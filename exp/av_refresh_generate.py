"""Run the AV with the canonical single-marker prompt, but "refresh" the injected
activation into the residual stream of the OUTPUT tokens during generation.

Motivation: the canonical AV sees the activation exactly once, at the single
`<concept>㈜</concept>` marker in the prompt. As generation proceeds the model may
"forget" / dilute that signal. Here we keep the standard prompt (so K=1, in-
distribution at the marker) and additionally ADD a decayed copy of the same
activation vector onto the embedding (layer-0 residual) of each generated token,
to keep the signal salient. Strength = alpha * decay(j) at output step j, where
the activation is L2-normalised to inj_scale (the same magnitude used at the
input marker). alpha=0 reproduces the canonical AV exactly (sanity anchor).

Decay schedule (j = 0-based output-token index):
    warm tokens at full strength, then exponential with timescale tau:
        s(j) = alpha                      for j < warm
        s(j) = alpha * exp(-(j-warm)/tau) for j >= warm
    Optionally hard-cut after --inject-steps (no injection past that).

Because the refresh must touch every decoded token, stock model.generate() won't
do — we run a manual, KV-cached, left-padded batched decode loop.

Output schema matches the warmstart/repeat parquets so score_repeat_ar.py can
consume it directly.
"""
from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm / target_scale).to(v.dtype)


def find_single_marker(ids, inj_id, left_id, right_id):
    """Return the single marker position, asserting canonical neighbours."""
    positions = [p for p in range(len(ids)) if ids[p] == inj_id]
    if len(positions) != 1:
        return None
    p = positions[0]
    if p - 1 < 0 or p + 1 >= len(ids):
        return None
    if ids[p - 1] != left_id or ids[p + 1] != right_id:
        return None
    return p


def decay_strength(j: int, alpha: float, warm: int, tau: float, inject_steps: int) -> float:
    """Refresh strength at 0-based output step j."""
    if inject_steps >= 0 and j >= inject_steps:
        return 0.0
    if j < warm:
        return alpha
    if tau <= 0:
        return 0.0
    return alpha * math.exp(-(j - warm) / tau)


@torch.inference_mode()
def generate_refresh_batch(prompt_embeds, prompt_masks, v_units, model, tok, embed_layer,
                           args, device):
    """Manual KV-cached batched decode with per-step residual refresh.

    prompt_embeds: list of [T_i, d] (activation already injected at the marker).
    prompt_masks:  list of [T_i]   (all ones; padding handled here).
    v_units:       list of [d] unit-norm activation vectors (per sample).
    Returns list of decoded strings (the generated continuation only).
    """
    d = prompt_embeds[0].shape[-1]
    B = len(prompt_embeds)
    lengths = [e.shape[0] for e in prompt_embeds]
    maxlen = max(lengths)

    emb = torch.zeros(B, maxlen, d, dtype=torch.bfloat16, device=device)
    att = torch.zeros(B, maxlen, dtype=torch.long, device=device)
    for i, (e, m) in enumerate(zip(prompt_embeds, prompt_masks)):
        L = e.shape[0]
        emb[i, maxlen - L:] = e.to(torch.bfloat16)
        att[i, maxlen - L:] = m
    V = torch.stack(v_units).to(device=device, dtype=torch.bfloat16)  # [B, d], unit norm

    pos = (att.cumsum(-1) - 1).clamp_min(0)  # [B, maxlen], left-pad aware
    out = model(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True)
    pkv = out.past_key_values
    logits = out.logits[:, -1, :]

    eos_id = tok.eos_token_id
    cur_pos = pos[:, -1:]  # [B, 1]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    gen_ids = [[] for _ in range(B)]
    refresh_mag = args.inj_scale  # added vector has norm = alpha*decay(j)*inj_scale

    for j in range(args.max_new_tokens):
        if args.temperature > 0:
            probs = torch.softmax(logits.float() / args.temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1).squeeze(-1)
        else:
            next_tok = logits.argmax(-1)
        next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
        for i in range(B):
            if not finished[i]:
                gen_ids[i].append(int(next_tok[i]))
        finished = finished | (next_tok == eos_id)
        if bool(finished.all()):
            break

        tok_emb = embed_layer(next_tok.unsqueeze(1)).to(torch.bfloat16)  # [B, 1, d]
        s = decay_strength(j, args.alpha, args.warm, args.tau, args.inject_steps)
        if s != 0.0:
            tok_emb = tok_emb + (s * refresh_mag) * V.unsqueeze(1)

        cur_pos = cur_pos + 1
        att = torch.cat([att, torch.ones(B, 1, dtype=torch.long, device=device)], dim=1)
        out = model(inputs_embeds=tok_emb, attention_mask=att, position_ids=cur_pos,
                    past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        logits = out.logits[:, -1, :]

    return [tok.decode(g, skip_special_tokens=True) for g in gen_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-ckpt", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    # refresh schedule
    ap.add_argument("--alpha", type=float, required=True,
                    help="peak refresh strength as a fraction of inj_scale (0 = canonical AV)")
    ap.add_argument("--warm", type=int, default=4, help="output tokens held at full strength")
    ap.add_argument("--tau", type=float, default=8.0, help="exponential decay timescale after warm")
    ap.add_argument("--inject-steps", type=int, default=-1,
                    help="hard cut: no refresh past this many output tokens (-1 = no cut)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    ckpt = Path(args.av_ckpt)
    meta = yaml.safe_load((ckpt / "nla_meta.yaml").read_text())
    args.inj_scale = float(meta["extraction"]["injection_scale"])
    d_model = int(meta["d_model"])
    av_template = meta["prompt_templates"]["av"]
    tm = meta["tokens"]
    inj_char = tm["injection_char"]
    inj_id = tm["injection_token_id"]
    left_id = tm["injection_left_neighbor_id"]
    right_id = tm["injection_right_neighbor_id"]
    print(f"[meta] d={d_model} inj_scale={args.inj_scale} inj_char={inj_char!r}(id={inj_id}) "
          f"alpha={args.alpha} warm={args.warm} tau={args.tau} inject_steps={args.inject_steps}")

    base = av_template.format(injection_char=inj_char)

    print(f"[load] {ckpt}")
    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def render(content):
        return tok.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt), torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    embed_layer = model.get_input_embeddings()

    text0 = render(base)
    ids0 = tok(text0, add_special_tokens=False)["input_ids"]
    marker0 = find_single_marker(ids0, inj_id, left_id, right_id)
    assert marker0 is not None, "canonical single marker not found / neighbours mismatch"
    print(f"[tok-check] marker at position {marker0}, neighbours OK")

    pr = pq.read_table(args.subset)
    sids = pr.column("sample_idx").to_pylist()
    acts = np.asarray(pr.column("activation").combine_chunks().values,
                      dtype=np.float32).reshape(len(sids), d_model)
    n = len(sids) if args.limit <= 0 else min(args.limit, len(sids))
    print(f"generating {n} samples (alpha={args.alpha}, batch_size={args.batch_size})")

    raw_out = [None] * n
    t0 = time.time()
    s = 0
    while s < n:
        idxs = list(range(s, min(s + args.batch_size, n)))
        prompt_embeds, prompt_masks, v_units, ok_idx = [], [], [], []
        for i in idxs:
            text = render(base)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            marker = find_single_marker(ids, inj_id, left_id, right_id)
            if marker is None:
                continue
            ids_t = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)
            e = embed_layer(ids_t).float()[0]
            v = torch.from_numpy(acts[i].copy()).to(args.device).float().view(1, -1)
            v_scaled = normalize_activation(v, args.inj_scale)[0]
            e[marker] = v_scaled
            v_unit = (v / v.norm().clamp_min(1e-12))[0]
            prompt_embeds.append(e)
            prompt_masks.append(torch.ones(e.shape[0], dtype=torch.long, device=args.device))
            v_units.append(v_unit)
            ok_idx.append(i)
        if prompt_embeds:
            outs = generate_refresh_batch(prompt_embeds, prompt_masks, v_units, model, tok,
                                          embed_layer, args, args.device)
            for i, o in zip(ok_idx, outs):
                raw_out[i] = o
        done = min(s + args.batch_size, n)
        if done % (args.batch_size * 4) == 0 or done == n:
            el = time.time() - t0
            print(f"  {done}/{n}  elapsed={el:.0f}s  ({done/el:.2f}/s)", flush=True)
            if s == 0 and outs:
                print(f"  sample[0] raw[:300]: {outs[0][:300]!r}")
        s += args.batch_size

    cleaned, parsed = [], []
    for r in raw_out:
        m = EXPLANATION_RE.search(r) if r else None
        if m:
            cleaned.append(m.group(1).strip()); parsed.append(True)
        else:
            cleaned.append(""); parsed.append(False)
    print(f"parsed: {sum(parsed)}/{n}")

    out = pa.table({
        "sample_idx": pa.array(sids[:n], type=pa.int64()),
        "warmstart_raw": pa.array([r or "" for r in raw_out]),
        "warmstart_cleaned": pa.array(cleaned),
        "warmstart_parsed": pa.array(parsed, type=pa.bool_()),
    })
    pq.write_table(out, args.out, compression="zstd")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
