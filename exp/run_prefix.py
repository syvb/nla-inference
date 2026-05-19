"""Prefix-prepending experiment: does padding `Final token: X` with prose-like
boilerplate help the AR reconstruct better?

All variants are `<prefix>Final token: X` (no quotes around X; colon kept).
Compare to plain swap_nq.

Prefixes tested:
  - p_dots4    : ".\n\n.\n\n"
  - p_nl4      : "\n\n\n\n"
  - p_phrase   : 'Structured format: a response.\n\nThe phrase "".\n\n.\n\n'
  - p_phrase2  : 'Structured format: a response.\n\nThe phrase refers to.\n\n.\n\n'
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
    mse_scale = float(meta["extraction"]["mse_scale"])
    template = meta["prompt_templates"]["ar"]
    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(str(ckpt), torch_dtype=dtype, trust_remote_code=True)
    backbone.lm_head = torch.nn.Identity()
    inner = backbone.model
    for attr in _FINAL_LN_ATTRS:
        if hasattr(inner, attr):
            setattr(inner, attr, torch.nn.Identity()); break
    d = backbone.config.hidden_size
    head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
    head.load_state_dict(load_file(str(ckpt / "value_head.safetensors")))
    return tok, backbone.to(device).eval(), head.to(device).eval(), mse_scale, template, d


def mse_nrm(p, g, s):
    pn = p / p.norm().clamp_min(1e-12) * s
    gn = g / g.norm().clamp_min(1e-12) * s
    return ((pn - gn) ** 2).mean().item()


@torch.inference_mode()
def reconstruct_batch(texts, tok, backbone, head, template, device):
    prompts = [template.format(explanation=t) for t in texts]
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
    h = backbone.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    return head(h[:, -1, :]).float().cpu()


def run_pass(name, texts, tok, backbone, head, template, device, batch_size, activations, mse_scale):
    n = len(texts)
    mse = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, batch_size):
        preds = reconstruct_batch(texts[i:i+batch_size], tok, backbone, head, template, device)
        for j, p in enumerate(preds):
            mse[i+j] = mse_nrm(p, torch.from_numpy(activations[i+j]), mse_scale)
    el = time.time() - t0
    print(f"[{name:20s}] {n} in {el:.1f}s  mse mean={mse.mean():.4f}  med={np.median(mse):.4f}", flush=True)
    return mse


# ─── Variants ────────────────────────────────────────────────────────
def with_prefix(prefix):
    def f(t): return f'{prefix}Final token: {t}'
    return f

PREFIXES = {
    "swap_nq":   "",
    "p_dots4":   ".\n\n.\n\n",
    "p_nl4":     "\n\n\n\n",
    "p_phrase":  'Structured format: a response.\n\nThe phrase "".\n\n.\n\n',
    "p_phrase2": 'Structured format: a response.\n\nThe phrase refers to.\n\n.\n\n',
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
    tok, backbone, head, mse_scale, template, d = load_critic(Path(args.ckpt), args.device)

    table = pq.read_table(args.parquet)
    n_total = table.num_rows
    n = min(args.limit, n_total)
    pick = rng.choice(n_total, size=n, replace=False); pick.sort()

    token_strs = [table.column("token_str")[int(i)].as_py() for i in pick]
    sample_idx = [table.column("sample_idx")[int(i)].as_py() for i in pick]
    act_flat = np.asarray(table.column("activation").combine_chunks().values).astype(np.float32).reshape(n_total, d)
    activations = act_flat[pick]
    rec_flat = np.asarray(table.column("recon").combine_chunks().values).astype(np.float32).reshape(n_total, d)
    recons = rec_flat[pick]

    base_mse = np.array(
        [mse_nrm(torch.from_numpy(recons[i]), torch.from_numpy(activations[i]), mse_scale) for i in range(n)],
        dtype=np.float32,
    )
    print(f"[baseline-parquet ] n={n}  mse mean={base_mse.mean():.4f} med={np.median(base_mse):.4f}")

    empty_mse = run_pass("empty", [""] * n, tok, backbone, head, template, args.device, args.batch_size, activations, mse_scale)
    results = {"base_mse_nrm": base_mse, "empty_mse_nrm": empty_mse}

    for name, prefix in PREFIXES.items():
        fn = with_prefix(prefix)
        texts = [fn(t) for t in token_strs]
        print(f"  example {name}: {texts[0]!r}")
        results[f"{name}_mse_nrm"] = run_pass(name, texts, tok, backbone, head, template, args.device, args.batch_size, activations, mse_scale)

    print()
    print("=" * 78)
    print(f"SUMMARY  ({args.parquet}, n={n})")
    print("=" * 78)
    print(f"  {'variant':28s}  mse_mean  mse_median  fve_over_empty")
    order = ["base"] + list(PREFIXES) + ["empty"]
    for name in order:
        m = results[f"{name}_mse_nrm"]
        f = ((empty_mse - m) / empty_mse).mean()
        print(f"  {name:28s}  {m.mean():.4f}    {np.median(m):.4f}      {f:+.4f}")

    out_cols = {
        "sample_idx": pa.array(sample_idx),
        "token_str": pa.array(token_strs),
    }
    for k, v in results.items():
        out_cols[k] = pa.array(v)
    out_cols["activation"] = pa.array(activations.tolist(), type=pa.list_(pa.float32()))
    pq.write_table(pa.table(out_cols), args.out, compression="zstd")
    print(f"[save] {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
