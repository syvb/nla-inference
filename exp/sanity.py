"""Three sanity checks on the AR input pipeline.

1. Tokenization: do prompt strings end in the expected critic-suffix tokens?
2. Padded-vs-unpadded equivalence: does batched left-padded inference give
   the same recon vector as single-example inference?
3. Variant strings: what are the exact tokens for each variant?

Also: compare against the NLACritic implementation in nla_inference.py to
confirm there's no semantic drift from my reimplementation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = Path("/home/ubuntu/ar_ckpt")
PARQUET = Path("/home/ubuntu/results_chat_20k.parquet")

# ─── Load critic exactly as my run_variants.py does ─────────────────────────

_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")


def load_critic(ckpt: Path, device: str, dtype=torch.bfloat16):
    meta = yaml.safe_load((ckpt / "nla_meta.yaml").read_text())
    mse_scale = float(meta["extraction"]["mse_scale"])
    template = meta["prompt_templates"]["ar"]
    suffix_ids = meta["tokens"]["critic_suffix_ids"]
    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        str(ckpt), torch_dtype=dtype, trust_remote_code=True
    )
    backbone.lm_head = torch.nn.Identity()
    inner = backbone.model
    for attr in _FINAL_LN_ATTRS:
        if hasattr(inner, attr):
            setattr(inner, attr, torch.nn.Identity()); break
    d = backbone.config.hidden_size
    head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
    head.load_state_dict(load_file(str(ckpt / "value_head.safetensors")))
    return tok, backbone.to(device).eval(), head.to(device).eval(), mse_scale, template, d, suffix_ids


def mse_nrm(p, g, s):
    pn = p / p.norm().clamp_min(1e-12) * s
    gn = g / g.norm().clamp_min(1e-12) * s
    return ((pn - gn) ** 2).mean().item()


# ─── Inference paths ────────────────────────────────────────────────────────

@torch.inference_mode()
def reconstruct_single_canonical(prompt, tok, backbone, head, device):
    """Mirrors NLACritic.reconstruct() from nla_inference.py — single example,
    no padding, no attention_mask, positional input_ids arg."""
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    h = backbone.model(ids, use_cache=False).last_hidden_state[0, -1]
    return head(h).float().cpu()


@torch.inference_mode()
def reconstruct_batched_leftpad(prompts, tok, backbone, head, device):
    """What run_variants.py does."""
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    h = backbone.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    return head(h[:, -1, :]).float().cpu()


# ─── Checks ─────────────────────────────────────────────────────────────────

def main():
    tok, backbone, head, mse_scale, template, d, suffix_ids = load_critic(CKPT, "cuda:0")

    print("=" * 78)
    print("CHECK 1: tokenization — last 5 tokens of prompts should match critic_suffix_ids")
    print("=" * 78)
    print(f"sidecar suffix_ids: {suffix_ids}")
    print(f"decoded:            {[tok.decode([i]) for i in suffix_ids]!r}")
    print()
    variants = {
        "swap": 'Final token: " disclaimer"',
        "no_colon": 'Final token " disclaimer"',
        "repeat10": ('Final token " disclaimer" ' * 9 + 'Final token " disclaimer"'),
        "phrased": 'Structured conversation: final token " disclaimer"',
        "real_explanation": "Structured FAQ format with a factual, informative tone signals a standard AI model description, establishing a definition of LaMDA.\n\nThe phrase \"It's been trained on a massive amount\" sets up a standard AI capability description.",
    }
    for name, expl in variants.items():
        prompt = template.format(explanation=expl)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        last5 = ids[-5:]
        match = "✓" if last5 == suffix_ids else "✗"
        print(f"  {name:18s}  n_tok={len(ids):4d}  last5={last5}  {match}")
        print(f"                      decoded last5: {[tok.decode([i]) for i in last5]!r}")

    print()
    print("=" * 78)
    print("CHECK 2: padded-vs-unpadded equivalence on real explanations")
    print("=" * 78)
    # Take 8 real rows from the parquet
    table = pq.read_table(str(PARQUET))
    n_total = table.num_rows
    expls = [table.column("explanation")[i].as_py() for i in range(8)]
    acts = np.asarray(table.column("activation").combine_chunks().values).astype(np.float32).reshape(n_total, d)[:8]
    parquet_recons = np.asarray(table.column("recon").combine_chunks().values).astype(np.float32).reshape(n_total, d)[:8]

    prompts = [template.format(explanation=e) for e in expls]

    # Single-example canonical path
    singles = []
    for p in prompts:
        singles.append(reconstruct_single_canonical(p, tok, backbone, head, "cuda:0"))
    singles = torch.stack(singles)  # [8, d]

    # Batched left-padded
    batched = reconstruct_batched_leftpad(prompts, tok, backbone, head, "cuda:0")  # [8, d]

    print(f"  {'i':>3s}  {'||single||':>10s} {'||batched||':>11s} {'||parq||':>9s} "
          f"{'cos(s,b)':>9s} {'cos(s,p)':>9s} {'cos(b,p)':>9s} {'mse_s':>8s} {'mse_b':>8s} {'mse_p':>8s}")
    for i in range(8):
        s = singles[i]
        b = batched[i]
        p = torch.from_numpy(parquet_recons[i])
        a = torch.from_numpy(acts[i])
        cos_sb = (s @ b / (s.norm() * b.norm())).item()
        cos_sp = (s @ p / (s.norm() * p.norm())).item()
        cos_bp = (b @ p / (b.norm() * p.norm())).item()
        print(f"  {i:>3d}  {s.norm().item():>10.3f} {b.norm().item():>11.3f} "
              f"{p.norm().item():>9.3f} {cos_sb:>9.6f} {cos_sp:>9.6f} {cos_bp:>9.6f} "
              f"{mse_nrm(s,a,mse_scale):>8.4f} {mse_nrm(b,a,mse_scale):>8.4f} {mse_nrm(p,a,mse_scale):>8.4f}")

    print()
    print("=" * 78)
    print("CHECK 3: rerun the swap variant on these 8 rows, both paths, compare")
    print("=" * 78)
    swap_prompts = [template.format(explanation=f'Final token: {json.dumps(table.column("token_str")[i].as_py())}') for i in range(8)]
    print(f"  example swap prompt (full):\n    {swap_prompts[0]!r}")
    singles_swap = torch.stack([reconstruct_single_canonical(p, tok, backbone, head, "cuda:0") for p in swap_prompts])
    batched_swap = reconstruct_batched_leftpad(swap_prompts, tok, backbone, head, "cuda:0")
    for i in range(8):
        s = singles_swap[i]; b = batched_swap[i]; a = torch.from_numpy(acts[i])
        cos_sb = (s @ b / (s.norm() * b.norm())).item()
        print(f"  i={i:>2d}  cos(single,batched)={cos_sb:.6f}  "
              f"mse_single={mse_nrm(s,a,mse_scale):.4f}  mse_batched={mse_nrm(b,a,mse_scale):.4f}")

    print()
    print("=" * 78)
    print("CHECK 4: vary batch composition — same row in two different batches")
    print("=" * 78)
    # Same prompt[0] in: alone, in batch of 8 (varying length), in batch of 8 (same length)
    a0 = swap_prompts[0]
    alone = reconstruct_batched_leftpad([a0], tok, backbone, head, "cuda:0")[0]
    with_long = reconstruct_batched_leftpad([a0] + prompts[1:8], tok, backbone, head, "cuda:0")[0]
    with_short = reconstruct_batched_leftpad([a0]*8, tok, backbone, head, "cuda:0")[0]
    print(f"  cos(alone, with_long_neighbors) = {(alone @ with_long / (alone.norm() * with_long.norm())).item():.6f}")
    print(f"  cos(alone, with_short_neighbors)= {(alone @ with_short / (alone.norm() * with_short.norm())).item():.6f}")

    print()
    print("=" * 78)
    print("CHECK 5: attention implementation in use")
    print("=" * 78)
    print(f"  backbone.config._attn_implementation: {getattr(backbone.config, '_attn_implementation', '?')}")


if __name__ == "__main__":
    main()
