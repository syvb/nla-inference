"""Run a configurable set of explanation variants through the AR critic.

Variants (each is a callable token_str -> explanation_text):
  - swap         : Final token: "X"          (baseline already covered, reproduce here)
  - no_colon     : Final token "X"
  - repeat10     : (Final token "X" ) * 10   (no colon, 10 copies, space-joined)
  - phrased      : Structured conversation: final token "X"

Same three controls as run_controls.py:
  - empty
  - wrong-token (no colon)

Used on both chat_20k and pretraining 'diverse' parquets.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")


def load_critic(ckpt: Path, device: str, dtype=torch.bfloat16):
    meta = yaml.safe_load((ckpt / "nla_meta.yaml").read_text())
    assert meta["role"] in ("critic", "ar")
    mse_scale = float(meta["extraction"]["mse_scale"])
    template = meta["prompt_templates"].get("ar") or meta["prompt_templates"]["critic"]
    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        str(ckpt), torch_dtype=dtype, trust_remote_code=True
    )
    backbone.lm_head = torch.nn.Identity()
    inner = backbone.model
    for attr in _FINAL_LN_ATTRS:
        if hasattr(inner, attr):
            setattr(inner, attr, torch.nn.Identity()); break
    else:
        raise AssertionError(f"no final-LN on {type(inner).__name__}")
    d = backbone.config.hidden_size
    head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
    head.load_state_dict(load_file(str(ckpt / "value_head.safetensors")))
    backbone = backbone.to(device).eval()
    head = head.to(device).eval()
    return tok, backbone, head, mse_scale, template, d


def mse_nrm(pred, gold, s):
    p = pred / pred.norm().clamp_min(1e-12) * s
    g = gold / gold.norm().clamp_min(1e-12) * s
    return ((p - g) ** 2).mean().item()


@torch.inference_mode()
def reconstruct_batch(texts, tok, backbone, head, template, device):
    prompts = [template.format(explanation=t) for t in texts]
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    h = backbone.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    return head(h[:, -1, :]).float().cpu()


def run_pass(name, texts, tok, backbone, head, template, device, batch_size,
             activations, mse_scale):
    n = len(texts)
    mse = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        preds = reconstruct_batch(batch, tok, backbone, head, template, device)
        for j, p in enumerate(preds):
            mse[i + j] = mse_nrm(p, torch.from_numpy(activations[i + j]), mse_scale)
    elapsed = time.time() - t0
    print(f"[{name:24s}] {n} in {elapsed:.1f}s ({n/elapsed:.0f}/s)  "
          f"mse mean={mse.mean():.4f}  median={np.median(mse):.4f}  "
          f"p25={np.percentile(mse,25):.4f}  p75={np.percentile(mse,75):.4f}",
          flush=True)
    return mse


# ─── Variants ──────────────────────────────────────────────────────────

def variant_swap(t): return f'Final token: {json.dumps(t)}'
def variant_no_colon(t): return f'Final token {json.dumps(t)}'
def variant_repeat10(t):
    one = f'Final token {json.dumps(t)}'
    return (one + ' ') * 9 + one    # 10 copies, space-joined
def variant_phrased(t): return f'Structured conversation: final token {json.dumps(t)}'

VARIANTS = {
    "swap": variant_swap,
    "no_colon": variant_no_colon,
    "repeat10": variant_repeat10,
    "phrased": variant_phrased,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[load] critic")
    tok, backbone, head, mse_scale, template, d = load_critic(Path(args.ckpt), args.device)

    print(f"[load] parquet {args.parquet}")
    table = pq.read_table(args.parquet)
    n_total = table.num_rows
    # Random subsample of size limit (reproducible)
    n = min(args.limit, n_total)
    pick = rng.choice(n_total, size=n, replace=False)
    pick.sort()
    print(f"[load] {n_total} rows; sampled {n}")

    explanations = [table.column("explanation")[int(i)].as_py() for i in pick]
    token_strs = [table.column("token_str")[int(i)].as_py() for i in pick]
    sample_idx = [table.column("sample_idx")[int(i)].as_py() for i in pick]

    # Activation / recon are list<float>; combine_chunks then slice.
    act_flat = np.asarray(table.column("activation").combine_chunks().values).astype(np.float32).reshape(n_total, d)
    activations = act_flat[pick]
    rec_flat = np.asarray(table.column("recon").combine_chunks().values).astype(np.float32).reshape(n_total, d)
    recons = rec_flat[pick]
    del act_flat, rec_flat

    # Baseline from existing recons
    base_mse = np.array(
        [mse_nrm(torch.from_numpy(recons[i]), torch.from_numpy(activations[i]), mse_scale)
         for i in range(n)],
        dtype=np.float32,
    )
    print(f"[baseline-parquet] mse mean={base_mse.mean():.4f} median={np.median(base_mse):.4f}")

    # Empty + wrong-token-no-colon controls
    empty_mse = run_pass("empty", [""] * n, tok, backbone, head, template,
                         args.device, args.batch_size, activations, mse_scale)
    perm = rng.permutation(n)
    for i in range(n):
        if perm[i] == i:
            perm[i] = (i + 1) % n
    wrong_texts = [variant_no_colon(token_strs[perm[i]]) for i in range(n)]
    wrong_mse = run_pass("wrong_no_colon", wrong_texts, tok, backbone, head, template,
                         args.device, args.batch_size, activations, mse_scale)

    results = {"base_mse_nrm": base_mse, "empty_mse_nrm": empty_mse,
               "wrong_no_colon_mse_nrm": wrong_mse}

    for name, fn in VARIANTS.items():
        texts = [fn(t) for t in token_strs]
        if name == "swap":  # show examples once
            print("  examples:")
            for s in texts[:3]:
                print("   ", repr(s)[:200])
        else:
            print(f"  {name} example: {texts[0]!r}")
        mse = run_pass(name, texts, tok, backbone, head, template,
                       args.device, args.batch_size, activations, mse_scale)
        results[f"{name}_mse_nrm"] = mse

    print()
    print("=" * 78)
    print(f"SUMMARY  ({args.parquet}, n={n})")
    print("=" * 78)
    print(f"  {'variant':28s}  mse_mean  mse_median  fve_over_empty")
    for name in ["base", "swap", "no_colon", "repeat10", "phrased",
                 "wrong_no_colon", "empty"]:
        m = results[f"{name}_mse_nrm"]
        f = ((empty_mse - m) / empty_mse).mean()
        print(f"  {name:28s}  {m.mean():.4f}    {np.median(m):.4f}      {f:+.4f}")

    out_cols = {
        "sample_idx": pa.array(sample_idx),
        "token_str": pa.array(token_strs),
        "explanation": pa.array(explanations),
    }
    for k, v in results.items():
        out_cols[k] = pa.array(v)
    out_cols["activation"] = pa.array(activations.tolist(), type=pa.list_(pa.float32()))
    pq.write_table(pa.table(out_cols), args.out, compression="zstd")
    print(f"[save] {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
