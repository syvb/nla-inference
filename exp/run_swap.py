"""Final-token-swap experiment for NLA reconstruction.

Loads the chat_20k results parquet; for each row, replaces the AV's
explanation with `Final token: "{json_escaped_token}"`, runs the AR critic
to reconstruct, and reports NMSE (mse_nrm = MSE of vectors normalized to
L2=sqrt(d), equals 2*(1-cos)) vs the baseline reconstruction already in
the parquet.

Output parquet contains both new recon vectors and metrics so the user can
re-analyse later.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Critic loading (copied from nla_inference.NLACritic, trimmed) ──────────

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
            setattr(inner, attr, torch.nn.Identity())
            break
    else:
        raise AssertionError(f"no final-LN on {type(inner).__name__}")

    d = backbone.config.hidden_size
    head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
    head.load_state_dict(load_file(str(ckpt / "value_head.safetensors")))

    backbone = backbone.to(device).eval()
    head = head.to(device).eval()
    return tok, backbone, head, mse_scale, template, d


# ─── Metrics ─────────────────────────────────────────────────────────────────

def mse_nrm(pred: torch.Tensor, gold: torch.Tensor, mse_scale: float) -> float:
    """README's mse_nrm: both vectors L2-normalised to mse_scale (= √d), per-elem MSE.
    Equals 2*(1-cos). Range [0, 4]. Orthogonal = 2."""
    p = pred / pred.norm().clamp_min(1e-12) * mse_scale
    g = gold / gold.norm().clamp_min(1e-12) * mse_scale
    return ((p - g) ** 2).mean().item()


def cos_sim(pred: torch.Tensor, gold: torch.Tensor) -> float:
    return (pred @ gold / (pred.norm() * gold.norm()).clamp_min(1e-12)).item()


# ─── Batched reconstruction ─────────────────────────────────────────────────

@torch.inference_mode()
def reconstruct_batch(
    texts: list[str], tok, backbone, head, template: str, device: str,
) -> torch.Tensor:
    """Returns [B, d] fp32 cpu tensor of reconstructed vectors."""
    prompts = [template.format(explanation=t) for t in texts]
    # Left-pad so last-token extraction at index -1 is the real last token.
    tok.padding_side = "left"
    pad = tok.pad_token_id
    if pad is None:
        tok.pad_token = tok.eos_token
        pad = tok.pad_token_id
    enc = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    h = backbone.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    last = h[:, -1, :]  # left-padded → last position is always real
    pred = head(last)
    return pred.float().cpu()


# ─── Main ─────────────────────────────────────────────────────────────────

def js_escape(s: str) -> str:
    """JSON.stringify-style escaping. json.dumps quotes + escapes \\, \", \\n, etc."""
    return json.dumps(s)  # includes the surrounding double-quotes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="AR critic checkpoint dir")
    ap.add_argument("--parquet", required=True, help="input results parquet")
    ap.add_argument("--out", required=True, help="output parquet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--verify-baseline", type=int, default=64,
                    help="re-reconstruct N rows from existing explanations "
                         "and confirm match to parquet's recon column")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    print(f"[load] critic {args.ckpt}")
    tok, backbone, head, mse_scale, template, d = load_critic(Path(args.ckpt), args.device)
    print(f"[load] d_model={d}  mse_scale={mse_scale:.3f}  "
          f"template={template!r}")

    print(f"[load] parquet {args.parquet}")
    table = pq.read_table(args.parquet)
    n_total = table.num_rows
    n = n_total if args.limit == 0 else min(args.limit, n_total)
    print(f"[load] {n_total} rows; using {n}")

    # Pull columns we need into python lists/arrays.
    explanations = table.column("explanation").to_pylist()[:n]
    token_strs = table.column("token_str").to_pylist()[:n]
    av_parsed = table.column("av_parsed").to_pylist()[:n]

    def _flatten_list_col(col, d):
        # ChunkedArray of list<float>; combine_chunks then values() gives the
        # underlying flat float buffer.
        arr = col.combine_chunks()
        return np.asarray(arr.values).astype(np.float32).reshape(-1, d)

    activations = _flatten_list_col(table.column("activation"), d)[:n]
    recons = _flatten_list_col(table.column("recon"), d)[:n]

    # ─── Baseline NMSE from existing recon column ─────────────────────────
    base_mse = np.zeros(n, dtype=np.float32)
    base_cos = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = torch.from_numpy(activations[i])
        r = torch.from_numpy(recons[i])
        base_mse[i] = mse_nrm(r, a, mse_scale)
        base_cos[i] = cos_sim(r, a)
    print(f"[baseline-from-parquet] mse_nrm mean={base_mse.mean():.4f} "
          f"median={np.median(base_mse):.4f}  cos mean={base_cos.mean():.4f}")

    # ─── Sanity-check that running the critic fresh matches the parquet ──
    if args.verify_baseline > 0:
        k = min(args.verify_baseline, n)
        print(f"[verify] re-reconstructing first {k} rows from explanations…")
        max_diff = 0.0
        max_mse_delta = 0.0
        for i in range(0, k, args.batch_size):
            batch = explanations[i : i + args.batch_size]
            preds = reconstruct_batch(batch, tok, backbone, head, template, args.device)
            for j, p in enumerate(preds):
                idx = i + j
                gold_recon = torch.from_numpy(recons[idx])
                # The parquet's recon was produced under maybe-different dtype/device;
                # cos should be >0.999 if everything matches.
                d_diff = (p - gold_recon).norm().item() / gold_recon.norm().clamp_min(1e-6).item()
                max_diff = max(max_diff, d_diff)
                fresh_mse = mse_nrm(p, torch.from_numpy(activations[idx]), mse_scale)
                max_mse_delta = max(max_mse_delta, abs(fresh_mse - base_mse[idx]))
        print(f"[verify] max recon-vector rel-diff={max_diff:.4g}  "
              f"max mse_nrm delta={max_mse_delta:.4g}")

    # ─── Swap experiment: explanation := `Final token: "{escaped}"` ───────
    print(f"[swap] building prompts and reconstructing…")
    swap_texts = [f"Final token: {js_escape(t)}" for t in token_strs]
    print(f"[swap] examples:")
    for s in swap_texts[:5]:
        print("  ", repr(s))

    swap_recons = np.zeros((n, d), dtype=np.float32)
    swap_mse = np.zeros(n, dtype=np.float32)
    swap_cos = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, args.batch_size):
        batch = swap_texts[i : i + args.batch_size]
        preds = reconstruct_batch(batch, tok, backbone, head, template, args.device)
        for j, p in enumerate(preds):
            idx = i + j
            swap_recons[idx] = p.numpy()
            a = torch.from_numpy(activations[idx])
            swap_mse[idx] = mse_nrm(p, a, mse_scale)
            swap_cos[idx] = cos_sim(p, a)
        if (i // args.batch_size) % args.progress_every == 0:
            elapsed = time.time() - t0
            done = i + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else 0
            print(f"[swap] {done}/{n}  rate={rate:.1f}/s  eta={eta:.0f}s  "
                  f"current swap_mse mean={swap_mse[:done].mean():.4f}")

    elapsed = time.time() - t0
    print(f"[swap] done in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    # ─── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"n = {n}  (av_parsed=True: {sum(av_parsed)})")
    print(f"  baseline mse_nrm:  mean={base_mse.mean():.4f}  median={np.median(base_mse):.4f}  "
          f"p25={np.percentile(base_mse, 25):.4f}  p75={np.percentile(base_mse, 75):.4f}")
    print(f"  swap     mse_nrm:  mean={swap_mse.mean():.4f}  median={np.median(swap_mse):.4f}  "
          f"p25={np.percentile(swap_mse, 25):.4f}  p75={np.percentile(swap_mse, 75):.4f}")
    print(f"  delta    (swap-base): mean={(swap_mse-base_mse).mean():.4f}  "
          f"median={np.median(swap_mse-base_mse):.4f}")
    # Variance-explained framing: how much does each recover from the
    # orthogonal-baseline (mse=2)?
    base_ve = (2.0 - base_mse) / 2.0
    swap_ve = (2.0 - swap_mse) / 2.0
    print(f"  baseline ve-vs-orth:  mean={base_ve.mean():.4f}  median={np.median(base_ve):.4f}")
    print(f"  swap     ve-vs-orth:  mean={swap_ve.mean():.4f}  median={np.median(swap_ve):.4f}")
    print(f"  swap/baseline ve ratio (mean of ratio): "
          f"{(swap_ve.clip(0) / base_ve.clip(1e-6)).mean():.4f}")

    # ─── Save ────────────────────────────────────────────────────────────
    out_table = pa.table({
        "sample_idx": pa.array(table.column("sample_idx").to_pylist()[:n]),
        "token_str": pa.array(token_strs),
        "explanation": pa.array(explanations),
        "swap_text": pa.array(swap_texts),
        "av_parsed": pa.array(av_parsed),
        "base_mse_nrm": pa.array(base_mse),
        "base_cos": pa.array(base_cos),
        "swap_mse_nrm": pa.array(swap_mse),
        "swap_cos": pa.array(swap_cos),
        "activation": pa.array(activations.tolist(), type=pa.list_(pa.float32())),
        "base_recon": pa.array(recons.tolist(), type=pa.list_(pa.float32())),
        "swap_recon": pa.array(swap_recons.tolist(), type=pa.list_(pa.float32())),
    })
    pq.write_table(out_table, args.out, compression="zstd")
    print(f"[save] wrote {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
