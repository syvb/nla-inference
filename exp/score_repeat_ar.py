"""Score AV-repeat variant parquets with the AR critic.

Takes the gold activation directly from the 2k subset parquet (sample_idx +
activation), so no large activations/results parquet upload is needed. For each
variant parquet (warmstart_cleaned + warmstart_parsed + sample_idx, same order
as the subset) computes mse_warmstart per row and writes {sample_idx,
mse_warmstart, warmstart_parsed}.
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
        attn_implementation="eager",
    )
    backbone.lm_head = torch.nn.Identity()
    inner = backbone.model
    for attr in _FINAL_LN_ATTRS:
        if hasattr(inner, attr):
            setattr(inner, attr, torch.nn.Identity()); break
    d = backbone.config.hidden_size
    head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
    head.load_state_dict(load_file(str(ckpt / "value_head.safetensors")))
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
    tok.padding_side = "left"; tok.truncation_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=True,
              truncation=True, max_length=max_length)
    ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
    h = backbone.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    last = h[:, -1, :].to(head.weight.device)
    return head(last).float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subset", required=True, help="repeat_subset_2k.parquet (sample_idx + activation gold)")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    sub = pq.read_table(args.subset)
    sub_sids = sub.column("sample_idx").to_pylist()
    d_guess = None
    acts = np.asarray(sub.column("activation").combine_chunks().values, dtype=np.float32)
    gold_by_sid = {}

    print(f"[load] critic {args.ckpt}")
    tok, backbone, head, mse_scale, template, d = load_critic(Path(args.ckpt), args.device)
    print(f"  d={d}  mse_scale={mse_scale:.4f}")
    acts = acts.reshape(len(sub_sids), d)
    for i, s in enumerate(sub_sids):
        gold_by_sid[int(s)] = acts[i]

    summary = []
    for var_path in args.variants:
        name = Path(var_path).stem
        t = pq.read_table(var_path)
        sids = t.column("sample_idx").to_pylist()
        texts = t.column("warmstart_cleaned").to_pylist()
        parsed = t.column("warmstart_parsed").to_pylist()
        n = len(sids)
        valid = [bool(p) and bool(tx) and int(s) in gold_by_sid
                 for p, tx, s in zip(parsed, texts, sids)]
        mse = np.full(n, np.nan, dtype=np.float32)
        vidx = [i for i, v in enumerate(valid) if v]
        t0 = time.time()
        for s0 in range(0, len(vidx), args.batch_size):
            chunk = vidx[s0:s0 + args.batch_size]
            preds = reconstruct_batch([texts[i] for i in chunk], tok, backbone, head,
                                      template, args.device, args.max_length)
            for j, i in enumerate(chunk):
                g = torch.from_numpy(gold_by_sid[int(sids[i])])
                mse[i] = mse_nrm(preds[j], g, mse_scale)
        el = time.time() - t0
        vv = mse[~np.isnan(mse)]
        print(f"[{name}] n_valid={len(vv)}/{n}  {el:.0f}s  mse mean={vv.mean():.4f}  med={np.median(vv):.4f}", flush=True)
        out = pa.table({
            "sample_idx": pa.array(sids, type=pa.int64()),
            "mse_warmstart": pa.array(mse),
            "warmstart_parsed": pa.array(parsed, type=pa.bool_()),
        })
        op = f"{args.out_dir}/{name}_scored.parquet"
        pq.write_table(out, op, compression="zstd")
        print(f"  wrote {op}")
        summary.append((name, vv.mean(), np.median(vv), len(vv)))

    print("\n=== SUMMARY (mse_nrm, lower=better) ===")
    for name, mm, md, nv in summary:
        print(f"  {name:30s}  mean={mm:.4f}  med={md:.4f}  n={nv}")


if __name__ == "__main__":
    main()
