"""Few-shot AV generation with real injected activations, optionally K-repeated.

Builds a multi-turn prompt:
  [user: AV template w/ K <concept> blocks]  [assistant: <explanation>shot1</explanation>]
  ... n_shots demonstrations, each with that shot's REAL activation injected ...
  [user: AV template w/ K <concept> blocks]  -> model generates for the REAL activation

Every user turn (shots + query) uses the same K-repeated `tags` layout, so the
shots demonstrate the exact format the query uses. The shot activations are real
activations (from the few-shot pool) injected into the shot turns; their
assistant text is the canonical AV explanation for that activation.

Because the token sequence is identical across all eval samples (only the
injected vectors differ, and only at the LAST K marker positions), we tokenize
once, inject the constant shot activations once, and per-sample overwrite only
the final K markers. Output schema matches the warmstart parquets.
"""
from __future__ import annotations

import argparse
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


def find_tag_markers(ids, inj_id, left_id, right_id):
    """All marker positions (tags mode): each marker bounded by left_id/right_id."""
    out = []
    for p in range(len(ids)):
        if ids[p] == inj_id:
            if p - 1 < 0 or p + 1 >= len(ids) or ids[p - 1] != left_id or ids[p + 1] != right_id:
                return None
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-ckpt", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fewshot-pool", required=True, help="parquet: sample_idx, explanation, activation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-repeat", type=int, required=True, help="K markers per <concept> turn (tags mode)")
    ap.add_argument("--n-shots", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    K = args.n_repeat
    ckpt = Path(args.av_ckpt)
    meta = yaml.safe_load((ckpt / "nla_meta.yaml").read_text())
    inj_scale = float(meta["extraction"]["injection_scale"])
    d_model = int(meta["d_model"])
    av_template = meta["prompt_templates"]["av"]
    tm = meta["tokens"]
    inj_char = tm["injection_char"]; inj_id = tm["injection_token_id"]
    left_id = tm["injection_left_neighbor_id"]; right_id = tm["injection_right_neighbor_id"]
    print(f"[meta] d={d_model} inj_scale={inj_scale} K={K} n_shots={args.n_shots}")

    # User-turn content: K separate <concept>㈜</concept> blocks (tags mode), adjacent.
    tag_unit = f"<concept>{inj_char}</concept>"
    base = av_template.format(injection_char=inj_char)
    assert base.count(tag_unit) == 1
    base = base.replace(tag_unit, "".join([tag_unit] * K))

    # Few-shot pool
    fp = pq.read_table(args.fewshot_pool)
    fs_expl = fp.column("explanation").to_pylist()[:args.n_shots]
    fs_act = np.asarray(fp.column("activation").combine_chunks().values, np.float32).reshape(-1, d_model)[:args.n_shots]
    assert len(fs_expl) == args.n_shots, f"pool has {len(fs_expl)} shots, need {args.n_shots}"

    def wrap(expl):
        expl = EXPLANATION_RE.sub(lambda m: m.group(1), expl).strip()  # de-tag if needed
        return f"<explanation>\n{expl}\n</explanation>"

    messages = []
    for i in range(args.n_shots):
        messages.append({"role": "user", "content": base})
        messages.append({"role": "assistant", "content": wrap(fs_expl[i])})
    messages.append({"role": "user", "content": base})

    print(f"[load] {ckpt}")
    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt), torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    embed_layer = model.get_input_embeddings()

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    markers = find_tag_markers(ids, inj_id, left_id, right_id)
    expect = (args.n_shots + 1) * K
    assert markers is not None and len(markers) == expect, (
        f"expected {expect} markers, got {None if markers is None else len(markers)}")
    # Group into n_shots+1 contiguous runs of K (turn order = marker order).
    groups = [markers[g * K:(g + 1) * K] for g in range(args.n_shots + 1)]
    query_group = groups[-1]
    print(f"[tok-check] prompt_tokens={len(ids)}  markers={len(markers)} "
          f"({args.n_shots+1} groups of {K})  query_markers={query_group}")

    # Build the prefix embeds with the constant shot activations injected once.
    ids_t = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)
    with torch.no_grad():
        prefix = embed_layer(ids_t).float()[0]  # [T, d]
    for g in range(args.n_shots):
        v = torch.from_numpy(fs_act[g]).to(args.device).float().view(1, -1)
        vs = normalize_activation(v, inj_scale)[0]
        for p in groups[g]:
            prefix[p] = vs
    prefix = prefix.to(torch.bfloat16)  # [T,d], query markers still hold raw marker embedding

    # Eval subset
    pr = pq.read_table(args.subset)
    sids = pr.column("sample_idx").to_pylist()
    acts = np.asarray(pr.column("activation").combine_chunks().values, np.float32).reshape(len(sids), d_model)
    n = len(sids) if args.limit <= 0 else min(args.limit, len(sids))
    T = prefix.shape[0]
    print(f"generating {n} samples (few-shot K={K}, prompt_tokens={T}, batch={args.batch_size})")

    raw_out = [None] * n
    t0 = time.time()

    @torch.inference_mode()
    def run(batch_vecs):
        B = len(batch_vecs)
        emb = prefix.unsqueeze(0).repeat(B, 1, 1).clone()  # [B,T,d]
        for b, vs in enumerate(batch_vecs):
            for p in query_group:
                emb[b, p] = vs
        att = torch.ones(B, T, dtype=torch.long, device=args.device)
        gen = model.generate(inputs_embeds=emb, attention_mask=att,
                             max_new_tokens=args.max_new_tokens,
                             do_sample=(args.temperature > 0), temperature=args.temperature,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return [tok.decode(r, skip_special_tokens=True) for r in gen]

    for s in range(0, n, args.batch_size):
        idxs = list(range(s, min(s + args.batch_size, n)))
        vecs = []
        for i in idxs:
            v = torch.from_numpy(acts[i]).to(args.device).float().view(1, -1)
            vecs.append(normalize_activation(v, inj_scale)[0].to(torch.bfloat16))
        outs = run(vecs)
        for i, o in zip(idxs, outs):
            raw_out[i] = o
        done = min(s + args.batch_size, n)
        if done % (args.batch_size * 4) == 0 or done == n:
            el = time.time() - t0
            print(f"  {done}/{n}  elapsed={el:.0f}s  ({done/el:.2f}/s)", flush=True)
            if s == 0:
                print(f"  sample[0] raw[:300]: {outs[0][:300]!r}")

    cleaned, parsed = [], []
    for r in raw_out:
        m = EXPLANATION_RE.search(r) if r else None
        if m: cleaned.append(m.group(1).strip()); parsed.append(True)
        else: cleaned.append(""); parsed.append(False)
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
