"""Run the AV model with the activation injected into K repeated marker tokens.

Idea: instead of the canonical single injection site `<concept>㈜</concept>`,
render `<concept>㈜㈜...㈜</concept>` with K copies of the marker and overwrite
ALL K marker-token embeddings with the SAME (L2-normalized, scaled) activation
vector. Still one <concept>...</concept> tag pair. K=1 reproduces the canonical
AV exactly (sanity anchor). This is OOD — the AV was trained on one marker — so
the question is whether the repeated signal helps the model attend to it.

Batched, left-padded `inputs_embeds` generation. Output schema matches the
sonnet/warmstart parquets so warmstart_run_ar_multi.py can consume it.
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


def find_marker_run(ids, inj_id, left_id, right_id, k, mode="markers"):
    """Return the K marker positions to inject, or None on mismatch.

    mode="markers": K copies of the marker inside ONE <concept> tag pair, i.e. a
      contiguous run `<concept>㈜㈜…㈜</concept>` — only the run as a whole is
      bounded by left_id (before) / right_id (after).
    mode="tags": K separate `<concept>㈜</concept>` blocks — each marker is
      individually bounded by left_id / right_id.
    """
    positions = [p for p in range(len(ids)) if ids[p] == inj_id]
    if len(positions) != k:
        return None
    if mode == "markers":
        if positions != list(range(positions[0], positions[0] + k)):
            return None
        p0, p1 = positions[0], positions[-1]
        if p0 - 1 < 0 or p1 + 1 >= len(ids):
            return None
        if ids[p0 - 1] != left_id or ids[p1 + 1] != right_id:
            return None
    else:  # tags: each marker bounded by its own <concept>…</concept>
        for p in positions:
            if p - 1 < 0 or p + 1 >= len(ids):
                return None
            if ids[p - 1] != left_id or ids[p + 1] != right_id:
                return None
    return positions


@torch.inference_mode()
def generate_batch(batch_embeds, batch_masks, model, tok, max_new_tokens, temperature, device):
    """batch_embeds: list of [T_i, d] tensors (per-sample, already injected).
    Left-pad to max T, build attention mask, generate."""
    d = batch_embeds[0].shape[-1]
    lengths = [e.shape[0] for e in batch_embeds]
    maxlen = max(lengths)
    B = len(batch_embeds)
    emb = torch.zeros(B, maxlen, d, dtype=torch.bfloat16, device=device)
    att = torch.zeros(B, maxlen, dtype=torch.long, device=device)
    for i, (e, m) in enumerate(zip(batch_embeds, batch_masks)):
        L = e.shape[0]
        emb[i, maxlen - L:] = e.to(torch.bfloat16)
        att[i, maxlen - L:] = m
    gen = model.generate(
        inputs_embeds=emb, attention_mask=att,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0), temperature=temperature,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return [tok.decode(row, skip_special_tokens=True) for row in gen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-ckpt", required=True)
    ap.add_argument("--subset", required=True, help="repeat_subset_2k.parquet: sample_idx, decoded_full, activation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-repeat", type=int, required=True, help="K: number of repeated markers")
    ap.add_argument("--repeat-mode", choices=["markers", "tags"], default="markers",
                    help="markers: K markers in one <concept> pair; tags: K separate <concept>㈜</concept> blocks")
    ap.add_argument("--separator", default="", help="string between repeated <concept> blocks (tags mode)")
    ap.add_argument("--explain", action="store_true",
                    help="add a system message explaining that the activation is duplicated")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=24)
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
    inj_char = tm["injection_char"]
    inj_id = tm["injection_token_id"]
    left_id = tm["injection_left_neighbor_id"]
    right_id = tm["injection_right_neighbor_id"]
    print(f"[meta] d={d_model} inj_scale={inj_scale} inj_char={inj_char!r}(id={inj_id}) "
          f"K={K} mode={args.repeat_mode} explain={args.explain}")

    # Build the prompt.
    if args.repeat_mode == "markers":
        # K markers inside the single <concept> tag pair: <concept>㈜㈜…㈜</concept>
        base = av_template.format(injection_char=inj_char * K)
    else:
        # K separate <concept>㈜</concept> blocks, joined by --separator.
        tag_unit = f"<concept>{inj_char}</concept>"
        base = av_template.format(injection_char=inj_char)
        assert base.count(tag_unit) == 1, f"expected exactly one {tag_unit!r} in template"
        base = base.replace(tag_unit, args.separator.join([tag_unit] * K))

    # Optional system message explaining the duplication.
    sys_msg = None
    if args.explain:
        sys_msg = (
            f"The activation vector you must explain has been duplicated across {K} "
            f"identical <concept> blocks in the input below. All {K} copies are the "
            f"exact same vector; the repetition is only to make the signal more "
            f"salient. Explain the single underlying activation as usual.")

    print(f"[load] {ckpt}")
    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)

    # Some Gemma chat templates reject a `system` role; detect once and fall back
    # to prepending the note to the user turn.
    system_supported = True
    if sys_msg is not None:
        try:
            tok.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenize=False, add_generation_prompt=True)
        except Exception as e:
            system_supported = False
            print(f"[note] system role unsupported ({type(e).__name__}); merging into user turn")

    def render(content):
        if sys_msg is None:
            msgs = [{"role": "user", "content": content}]
        elif system_supported:
            msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": content}]
        else:
            msgs = [{"role": "user", "content": sys_msg + "\n\n" + content}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    print(f"[load model] {ckpt}")
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt), torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    embed_layer = model.get_input_embeddings()  # built-in √d scaling for Gemma3

    # Verify K markers tokenize as K inj_id tokens with correct neighbors.
    text0 = render(base)
    ids0 = tok(text0, add_special_tokens=False)["input_ids"]
    run0 = find_marker_run(ids0, inj_id, left_id, right_id, K, args.repeat_mode)
    assert run0 is not None, (
        f"K={K} ({args.repeat_mode}) did not tokenize as {K} inj_id tokens "
        f"bounded by left={left_id}/right={right_id}. "
        f"inj_id count={sum(1 for t in ids0 if t == inj_id)}")
    print(f"[tok-check] K={K} ({args.repeat_mode}): marker positions {run0}, neighbors OK")

    pr = pq.read_table(args.subset)
    sids = pr.column("sample_idx").to_pylist()
    decoded = pr.column("decoded_full").to_pylist()
    acts = np.asarray(pr.column("activation").combine_chunks().values, dtype=np.float32).reshape(len(sids), d_model)
    n = len(sids) if args.limit <= 0 else min(args.limit, len(sids))
    print(f"generating {n} samples (K={K}, batch_size={args.batch_size})")

    raw_out = [None] * n
    t0 = time.time()
    s = 0
    while s < n:
        idxs = list(range(s, min(s + args.batch_size, n)))
        batch_embeds, batch_masks, ok_idx = [], [], []
        for i in idxs:
            # Prompt content is identical across samples (the marker run); only
            # the injected vector differs. Render per-sample for safety.
            text = render(base)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            run = find_marker_run(ids, inj_id, left_id, right_id, K, args.repeat_mode)
            if run is None:
                continue
            ids_t = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)
            e = embed_layer(ids_t).float()[0]  # [T, d], already √d-scaled
            v = torch.from_numpy(acts[i]).to(args.device).float().view(1, -1)
            v_scaled = normalize_activation(v, inj_scale)[0]
            for p in run:
                e[p] = v_scaled
            batch_embeds.append(e)
            batch_masks.append(torch.ones(e.shape[0], dtype=torch.long, device=args.device))
            ok_idx.append(i)
        if batch_embeds:
            outs = generate_batch(batch_embeds, batch_masks, model, tok,
                                  args.max_new_tokens, args.temperature, args.device)
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
