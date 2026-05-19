"""Control variants for the final-token swap experiment.

Same setup as run_swap.py, but in addition to baseline (full explanation) and
swap (Final token: "X"), runs three controls:

  - empty:     explanation = ""
  - permuted:  explanations shuffled across rows (real text, wrong row)
  - wrong:     Final token: "<random other row's token>"

Compare the three controls to the actual swap to see how much the "final token"
text is actually doing vs. the AR's prior.
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
             activations, mse_scale, store_recons=False, d=None):
    n = len(texts)
    recons = np.zeros((n, d), dtype=np.float32) if store_recons else None
    mse = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        preds = reconstruct_batch(batch, tok, backbone, head, template, device)
        for j, p in enumerate(preds):
            idx = i + j
            mse[idx] = mse_nrm(p, torch.from_numpy(activations[idx]), mse_scale)
            if store_recons:
                recons[idx] = p.numpy()
    elapsed = time.time() - t0
    print(f"[{name}] {n} in {elapsed:.1f}s ({n/elapsed:.0f}/s)  "
          f"mse mean={mse.mean():.4f}  median={np.median(mse):.4f}  "
          f"p25={np.percentile(mse,25):.4f}  p75={np.percentile(mse,75):.4f}")
    return mse, recons


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

    print(f"[load] parquet")
    table = pq.read_table(args.parquet)
    n_total = table.num_rows
    n = min(args.limit, n_total)
    explanations = table.column("explanation").to_pylist()[:n]
    token_strs = table.column("token_str").to_pylist()[:n]
    av_parsed = table.column("av_parsed").to_pylist()[:n]
    sample_idx = table.column("sample_idx").to_pylist()[:n]
    activations = np.asarray(table.column("activation").combine_chunks().values).astype(np.float32).reshape(-1, d)[:n]
    recons_parquet = np.asarray(table.column("recon").combine_chunks().values).astype(np.float32).reshape(-1, d)[:n]

    # Baseline from existing parquet recons
    base_mse = np.array([
        mse_nrm(torch.from_numpy(recons_parquet[i]),
                torch.from_numpy(activations[i]), mse_scale)
        for i in range(n)
    ], dtype=np.float32)
    print(f"[baseline-parquet] mse mean={base_mse.mean():.4f} "
          f"median={np.median(base_mse):.4f}")

    # ─── Variants ──────────────────────────────────────────────────────
    swap_texts = [f"Final token: {json.dumps(t)}" for t in token_strs]

    empty_texts = ["" for _ in range(n)]

    perm = rng.permutation(n)
    # avoid self-mapping
    for i in range(n):
        if perm[i] == i:
            perm[i] = (i + 1) % n
    permuted_texts = [explanations[perm[i]] for i in range(n)]

    perm2 = rng.permutation(n)
    for i in range(n):
        if perm2[i] == i:
            perm2[i] = (i + 1) % n
    wrong_token_texts = [f"Final token: {json.dumps(token_strs[perm2[i]])}" for i in range(n)]

    swap_mse, swap_recons = run_pass("swap", swap_texts, tok, backbone, head, template,
                                     args.device, args.batch_size, activations, mse_scale,
                                     store_recons=True, d=d)
    empty_mse, _ = run_pass("empty", empty_texts, tok, backbone, head, template,
                            args.device, args.batch_size, activations, mse_scale, d=d)
    perm_mse, _ = run_pass("permuted", permuted_texts, tok, backbone, head, template,
                           args.device, args.batch_size, activations, mse_scale, d=d)
    wrong_mse, _ = run_pass("wrong-token", wrong_token_texts, tok, backbone, head, template,
                            args.device, args.batch_size, activations, mse_scale, d=d)

    print()
    print("=" * 70)
    print(f"SUMMARY n={n}  (av_parsed=True: {sum(av_parsed)})")
    print("=" * 70)
    print(f"  variant       mse_nrm       cos_equiv  fve(vs orth) fve(vs predict-mean)")
    pmean = 0.7335  # Var(v_nrm) on training set, per README
    for label, m in [("baseline (full explanation)", base_mse),
                     ("swap (Final token: <real>)", swap_mse),
                     ("wrong-token (Final token: <wrong>)", wrong_mse),
                     ("permuted (other row's explanation)", perm_mse),
                     ("empty ('')", empty_mse)]:
        mm = m.mean()
        cos = 1 - mm/2
        fve_orth = (2 - mm) / 2
        fve_pm = 1 - mm/pmean
        print(f"  {label:36s}  {mm:.4f}     {cos:.4f}     {fve_orth:.4f}     {fve_pm:.4f}")
    print()
    print("Per-row variance contribution decomposition (how much of the reconstruction")
    print("improvement over the orthogonal-baseline is due to the explanation vs the")
    print("AR's prior on the prompt template alone?):")
    rec_full = (2 - base_mse) / 2
    rec_swap = (2 - swap_mse) / 2
    rec_empty = (2 - empty_mse) / 2
    rec_wrong = (2 - wrong_mse) / 2
    contrib_token = (rec_swap - rec_empty) / np.clip(rec_full - rec_empty, 1e-6, None)
    contrib_wrong = (rec_wrong - rec_empty) / np.clip(rec_full - rec_empty, 1e-6, None)
    print(f"  fraction of (full-vs-empty) recovery from `Final token: <real>` alone:  "
          f"mean={contrib_token.mean():.4f} median={np.median(contrib_token):.4f}")
    print(f"  fraction of (full-vs-empty) recovery from `Final token: <wrong>` alone: "
          f"mean={contrib_wrong.mean():.4f} median={np.median(contrib_wrong):.4f}")

    out_table = pa.table({
        "sample_idx": pa.array(sample_idx),
        "token_str": pa.array(token_strs),
        "explanation": pa.array(explanations),
        "swap_text": pa.array(swap_texts),
        "wrong_token_text": pa.array(wrong_token_texts),
        "permuted_text": pa.array(permuted_texts),
        "av_parsed": pa.array(av_parsed),
        "base_mse_nrm": pa.array(base_mse),
        "swap_mse_nrm": pa.array(swap_mse),
        "wrong_token_mse_nrm": pa.array(wrong_mse),
        "permuted_mse_nrm": pa.array(perm_mse),
        "empty_mse_nrm": pa.array(empty_mse),
        "activation": pa.array(activations.tolist(), type=pa.list_(pa.float32())),
        "base_recon": pa.array(recons_parquet.tolist(), type=pa.list_(pa.float32())),
        "swap_recon": pa.array(swap_recons.tolist(), type=pa.list_(pa.float32())),
    })
    pq.write_table(out_table, args.out, compression="zstd")
    print(f"[save] {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
