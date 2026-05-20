"""Run AR critic on Sonnet warm-start explanations + compare to AV.

Inputs:
  --sonnet     sonnet46_500.parquet (sample_idx, warmstart_cleaned, warmstart_parsed)
  --activations activations_chat_20k.parquet (sample_idx → activation)
  --results    results_chat_20k.parquet (sample_idx → explanation_av, recon)
  --ckpt       AR critic checkpoint dir

For each sample:
  - mse_av_saved:  from results.recon vs activation (the trained AR's own output)
  - mse_av_rerun:  re-run AR on explanation → mse_nrm
  - mse_warmstart: re-run AR on warmstart_cleaned → mse_nrm  (None if not parsed)
  - mse_empty:     AR on "" → mse_nrm
"""
from __future__ import annotations

import argparse
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
    # device_map="auto" shards the model across all visible GPUs via accelerate.
    backbone = AutoModelForCausalLM.from_pretrained(
        str(ckpt), torch_dtype=dtype, trust_remote_code=True,
        device_map="auto" if device == "auto" else None,
    )
    backbone.lm_head = torch.nn.Identity()
    inner = backbone.model
    for attr in _FINAL_LN_ATTRS:
        if hasattr(inner, attr):
            setattr(inner, attr, torch.nn.Identity()); break
    d = backbone.config.hidden_size
    head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
    head.load_state_dict(load_file(str(ckpt / "value_head.safetensors")))
    if device == "auto":
        # Place head on the same device as the backbone's last layer output.
        head = head.to(next(backbone.parameters()).device).eval()
        backbone = backbone.eval()
    else:
        backbone = backbone.to(device).eval()
        head = head.to(device).eval()
    return tok, backbone, head, mse_scale, template, d


def mse_nrm(p, g, s):
    pn = p / p.norm().clamp_min(1e-12) * s
    gn = g / g.norm().clamp_min(1e-12) * s
    return ((pn - gn) ** 2).mean().item()


@torch.inference_mode()
def reconstruct_batch(texts, tok, backbone, head, template, device, max_length=2048):
    prompts = [template.format(explanation=t) for t in texts]
    tok.padding_side = "left"
    tok.truncation_side = "left"  # preserve the AR suffix tokens at the end
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=True,
              truncation=True, max_length=max_length)
    embed_device = backbone.get_input_embeddings().weight.device if device == "auto" else device
    ids = enc["input_ids"].to(embed_device); mask = enc["attention_mask"].to(embed_device)
    h = backbone.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    # h may be on the last layer's device (≠ head's device) under device_map=auto.
    last = h[:, -1, :].to(head.weight.device)
    return head(last).float().cpu()


def run_pass(name, texts, tok, backbone, head, template, device, batch_size, activations, mse_scale, valid_mask):
    n = len(texts)
    mse = np.full(n, np.nan, dtype=np.float32)
    t0 = time.time()
    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    for s in range(0, len(valid_idx), batch_size):
        chunk_idx = valid_idx[s:s+batch_size]
        chunk_texts = [texts[i] for i in chunk_idx]
        preds = reconstruct_batch(chunk_texts, tok, backbone, head, template, device)
        for j, i in enumerate(chunk_idx):
            mse[i] = mse_nrm(preds[j], torch.from_numpy(activations[i]), mse_scale)
    el = time.time() - t0
    valid_vals = mse[~np.isnan(mse)]
    print(f"[{name:20s}] n_valid={len(valid_vals)}/{n}  {el:.1f}s  "
          f"mse mean={valid_vals.mean():.4f}  med={np.median(valid_vals):.4f}", flush=True)
    return mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonnet", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    print(f"[load] sonnet      {args.sonnet}")
    sonnet_tbl = pq.read_table(args.sonnet)
    sample_idx = sonnet_tbl.column("sample_idx").to_pylist()
    n = len(sample_idx)
    warm_cleaned = sonnet_tbl.column("warmstart_cleaned").to_pylist()
    warm_parsed = sonnet_tbl.column("warmstart_parsed").to_pylist()
    print(f"  n={n}, parsed={sum(warm_parsed)}")

    print(f"[load] activations {args.activations}")
    act_tbl = pq.read_table(args.activations, columns=["sample_idx", "activation"])
    act_idx = np.asarray(act_tbl.column("sample_idx").to_pylist())
    pos = {int(s): i for i, s in enumerate(act_idx)}
    pick = [pos[int(s)] for s in sample_idx]

    # Activations are fixed_size_list[3840]
    print(f"[load] critic      {args.ckpt}")
    tok, backbone, head, mse_scale, template, d = load_critic(Path(args.ckpt), args.device)
    print(f"  d={d}  mse_scale={mse_scale:.4f}")

    act_flat = np.asarray(act_tbl.column("activation").combine_chunks().values).astype(np.float32).reshape(-1, d)
    activations = act_flat[pick]
    del act_flat

    print(f"[load] results     {args.results}")
    res_tbl = pq.read_table(args.results, columns=["sample_idx", "explanation", "recon", "av_parsed", "token_str"])
    res_idx = np.asarray(res_tbl.column("sample_idx").to_pylist())
    rpos = {int(s): i for i, s in enumerate(res_idx)}
    rpick = [rpos[int(s)] for s in sample_idx]
    explanations_av = [res_tbl.column("explanation")[i].as_py() for i in rpick]
    av_parsed = [res_tbl.column("av_parsed")[i].as_py() for i in rpick]
    token_strs = [res_tbl.column("token_str")[i].as_py() for i in rpick]
    rec_flat = np.asarray(res_tbl.column("recon").combine_chunks().values).astype(np.float32).reshape(-1, d)
    recons_av_saved = rec_flat[rpick]
    del rec_flat

    # mse_av_saved — from the saved recon vector
    print("[compute] mse_av_saved (from saved recon)")
    mse_av_saved = np.array(
        [mse_nrm(torch.from_numpy(recons_av_saved[i]), torch.from_numpy(activations[i]), mse_scale) for i in range(n)],
        dtype=np.float32,
    )
    valid_av_saved = ~np.isnan(mse_av_saved)
    print(f"  n_valid={valid_av_saved.sum()}/{n}  mean={mse_av_saved[valid_av_saved].mean():.4f}")

    # mse_av_rerun — re-run AR on the AV explanation (sanity vs saved)
    av_valid = [bool(p) and bool(e) for p, e in zip(av_parsed, explanations_av)]
    mse_av_rerun = run_pass(
        "av_rerun", [e or "" for e in explanations_av],
        tok, backbone, head, template, args.device, args.batch_size,
        activations, mse_scale, av_valid,
    )

    # mse_warmstart
    ws_valid = [bool(p) and bool(t) for p, t in zip(warm_parsed, warm_cleaned)]
    mse_warmstart = run_pass(
        "warmstart_sonnet46", warm_cleaned,
        tok, backbone, head, template, args.device, args.batch_size,
        activations, mse_scale, ws_valid,
    )

    # mse_empty
    mse_empty = run_pass(
        "empty", [""] * n, tok, backbone, head, template, args.device, args.batch_size,
        activations, mse_scale, [True] * n,
    )

    # Joint mask: rows where BOTH AV and warmstart parsed (for paired comparison)
    both_valid = np.array([av_valid[i] and ws_valid[i] for i in range(n)])
    print()
    print("=" * 78)
    print(f"SUMMARY  n_paired={both_valid.sum()}/{n}")
    print("=" * 78)
    print(f"  {'variant':24s}  mse_mean  mse_median  fve_over_empty")
    e = mse_empty
    for name, m in [
        ("av_saved",     mse_av_saved),
        ("av_rerun",     mse_av_rerun),
        ("warmstart_46", mse_warmstart),
        ("empty",        mse_empty),
    ]:
        mask = both_valid & ~np.isnan(m) & ~np.isnan(e)
        mv = m[mask]; ev = e[mask]
        f = ((ev - mv) / ev).mean() if mask.sum() else float("nan")
        print(f"  {name:24s}  {mv.mean():.4f}    {np.median(mv):.4f}      {f:+.4f}    (n={mask.sum()})")

    # Per-row paired delta
    if both_valid.sum():
        paired_delta = mse_warmstart[both_valid] - mse_av_rerun[both_valid]
        wins_av = (paired_delta > 0).sum()
        wins_ws = (paired_delta < 0).sum()
        print(f"  paired Δ (warm - av): mean={paired_delta.mean():+.4f}  median={np.median(paired_delta):+.4f}")
        print(f"  AV better on {wins_av}/{both_valid.sum()} rows; warm better on {wins_ws}")

    out_cols = {
        "sample_idx": pa.array(sample_idx, type=pa.int64()),
        "token_str": pa.array(token_strs),
        "explanation_av": pa.array(explanations_av),
        "warmstart_cleaned": pa.array(warm_cleaned),
        "av_parsed": pa.array(av_parsed, type=pa.bool_()),
        "warmstart_parsed": pa.array(warm_parsed, type=pa.bool_()),
        "mse_av_saved": pa.array(mse_av_saved),
        "mse_av_rerun": pa.array(mse_av_rerun),
        "mse_warmstart": pa.array(mse_warmstart),
        "mse_empty": pa.array(mse_empty),
        "activation": pa.array(activations.tolist(), type=pa.list_(pa.float32())),
    }
    pq.write_table(pa.table(out_cols), args.out, compression="zstd")
    print(f"[save] {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
