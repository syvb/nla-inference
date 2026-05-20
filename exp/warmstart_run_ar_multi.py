"""Run the AR critic on multiple warm-start variant parquets in one model load.

Reads activations + AV explanations once, then for each provided variant
parquet computes mse_warmstart and writes a result parquet. Saves model
load time vs running run_ar_long.py separately for each variant.
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
    tok.truncation_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=True,
              truncation=True, max_length=max_length)
    embed_device = backbone.get_input_embeddings().weight.device if device == "auto" else device
    ids = enc["input_ids"].to(embed_device); mask = enc["attention_mask"].to(embed_device)
    h = backbone.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    last = h[:, -1, :].to(head.weight.device)
    return head(last).float().cpu()


def run_pass(name, texts, tok, backbone, head, template, device, batch_size, activations, mse_scale, valid_mask, max_length):
    n = len(texts)
    mse = np.full(n, np.nan, dtype=np.float32)
    t0 = time.time()
    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    for s in range(0, len(valid_idx), batch_size):
        chunk_idx = valid_idx[s:s+batch_size]
        chunk_texts = [texts[i] for i in chunk_idx]
        preds = reconstruct_batch(chunk_texts, tok, backbone, head, template, device, max_length)
        for j, i in enumerate(chunk_idx):
            mse[i] = mse_nrm(preds[j], torch.from_numpy(activations[i]), mse_scale)
    el = time.time() - t0
    valid_vals = mse[~np.isnan(mse)]
    print(f"[{name:24s}] n_valid={len(valid_vals)}/{n}  {el:.1f}s  "
          f"mse mean={valid_vals.mean():.4f}  med={np.median(valid_vals):.4f}", flush=True)
    return mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True,
                    help="list of variant sonnet parquets to evaluate")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Anchor on the first variant for sample order
    print(f"[load] anchor variant: {args.variants[0]}")
    anchor = pq.read_table(args.variants[0])
    sample_idx = anchor.column("sample_idx").to_pylist()
    n = len(sample_idx)

    print(f"[load] activations {args.activations}")
    act_tbl = pq.read_table(args.activations, columns=["sample_idx", "activation"])
    act_idx = np.asarray(act_tbl.column("sample_idx").to_pylist())
    pos = {int(s): i for i, s in enumerate(act_idx)}
    pick = [pos[int(s)] for s in sample_idx]

    print(f"[load] critic {args.ckpt}")
    tok, backbone, head, mse_scale, template, d = load_critic(Path(args.ckpt), args.device)
    print(f"  d={d}  mse_scale={mse_scale:.4f}")

    act_flat = np.asarray(act_tbl.column("activation").combine_chunks().values).astype(np.float32).reshape(-1, d)
    activations = act_flat[pick]
    del act_flat

    print(f"[load] results {args.results}")
    res_tbl = pq.read_table(args.results, columns=["sample_idx", "recon"])
    res_idx = np.asarray(res_tbl.column("sample_idx").to_pylist())
    rpos = {int(s): i for i, s in enumerate(res_idx)}
    rpick = [rpos[int(s)] for s in sample_idx]
    rec_flat = np.asarray(res_tbl.column("recon").combine_chunks().values).astype(np.float32).reshape(-1, d)
    recons_av_saved = rec_flat[rpick]
    del rec_flat

    mse_av_saved = np.array(
        [mse_nrm(torch.from_numpy(recons_av_saved[i]), torch.from_numpy(activations[i]), mse_scale) for i in range(n)],
        dtype=np.float32,
    )
    print(f"[av_saved] mean={mse_av_saved.mean():.4f}")

    mse_empty = run_pass("empty", [""] * n, tok, backbone, head, template, args.device, args.batch_size,
                         activations, mse_scale, [True] * n, args.max_length)

    # For each variant
    summary = []
    for var_path in args.variants:
        var_name = Path(var_path).stem.replace("sonnet46_pt_", "")
        print(f"\n[variant] {var_name}")
        var_tbl = pq.read_table(var_path)
        v_sids = var_tbl.column("sample_idx").to_pylist()
        assert v_sids == sample_idx, f"sample_idx mismatch in {var_path}"
        texts = var_tbl.column("warmstart_cleaned").to_pylist()
        parsed = var_tbl.column("warmstart_parsed").to_pylist()
        valid = [bool(p) and bool(t) for p, t in zip(parsed, texts)]
        mse_v = run_pass(var_name, texts, tok, backbone, head, template,
                         args.device, args.batch_size, activations, mse_scale, valid, args.max_length)

        out_path = f"{args.out_dir}/warmstart_pt_{var_name}_out.parquet"
        out = pa.table({
            "sample_idx": pa.array(sample_idx, type=pa.int64()),
            "mse_av_saved": pa.array(mse_av_saved),
            "mse_warmstart": pa.array(mse_v),
            "mse_empty": pa.array(mse_empty),
            "warmstart_parsed": pa.array(parsed, type=pa.bool_()),
        })
        pq.write_table(out, out_path, compression="zstd")
        print(f"  saved {out_path}")
        # FVE
        mask = ~np.isnan(mse_v) & ~np.isnan(mse_empty)
        f = ((mse_empty[mask] - mse_v[mask]) / mse_empty[mask]).mean()
        summary.append((var_name, mse_v[mask].mean(), np.median(mse_v[mask]), f, mask.sum()))

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'variant':24s}  mse_mean  mse_med   FVE        n_valid")
    for name, mm, md, f, nv in summary:
        print(f"  {name:24s}  {mm:.4f}    {md:.4f}    {f:+.4f}    {nv}")


if __name__ == "__main__":
    main()
