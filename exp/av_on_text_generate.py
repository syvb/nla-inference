"""Run the trained AV as a plain text explainer (no activation injection).

Thread A, but with the AV model instead of Sonnet 4.6. We give the AV the
SOURCE TEXT (decoded_full prefix) in place of the injected activation, with a
minimally-modified prompt that says it's getting text rather than an activation
vector. Everything else about the AV prompt + <explanation> format is unchanged.

Batched left-padded HF generation for throughput.

Output schema matches the sonnet/warmstart parquets so warmstart_run_ar_multi.py
can consume it directly.
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


def build_text_prompt(av_template: str, inj_char: str) -> str:
    """Take the canonical AV prompt (which describes an activation vector) and
    minimally rewrite it to describe a TEXT SNIPPET instead. Returns a template
    string with a single `{source_text}` placeholder where the <concept> body
    goes.

    We do targeted string replacements rather than a hand-rewrite to stay as
    close to the trained prompt as possible.
    """
    t = av_template
    # The canonical template has the literal placeholder `{injection_char}`
    # inside the <concept> tags. Swap it for our {source_text} placeholder.
    assert "{injection_char}" in t, f"'{{injection_char}}' placeholder not in AV template"
    t = t.replace("{injection_char}", "{source_text}")
    # Minimal wording swaps: activation vector -> text snippet.
    repls = [
        ("activation vectors from a language model", "text snippets from a language model"),
        ("the semantic content of that activation vector", "the semantic content of that text snippet"),
        ("We will pass the vector enclosed", "We will pass the text enclosed"),
        ("produce an explanation for the vector", "produce an explanation for the text"),
        ("text snippets describing that vector", "text snippets describing that text"),
        ("Here is the vector:", "Here is the text:"),
    ]
    for a, b in repls:
        t = t.replace(a, b)
    return t


@torch.inference_mode()
def generate_batch(prompts, tok, model, device, max_new_tokens, temperature):
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # Render chat template per prompt, then batch-tokenize with left padding.
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=2048, add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    gen = model.generate(
        input_ids=ids, attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0), temperature=temperature,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    # New tokens are everything past the (padded) prompt length.
    new = gen[:, ids.shape[1]:]
    return [tok.decode(row, skip_special_tokens=True) for row in new]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-ckpt", required=True)
    ap.add_argument("--prompts", required=True, help="prompts parquet with decoded_full + sample_idx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = Path(args.av_ckpt)
    meta = yaml.safe_load((ckpt / "nla_meta.yaml").read_text())
    av_template = meta["prompt_templates"]["av"]
    inj_char = meta["tokens"]["injection_char"]

    text_template = build_text_prompt(av_template, inj_char)
    print("[prompt] text-mode AV template:")
    print(text_template.replace("{source_text}", "<<SOURCE TEXT HERE>>")[:700])
    print("...")

    print(f"[load] {ckpt}")
    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt), torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()

    pr = pq.read_table(args.prompts)
    all_sids = pr.column("sample_idx").to_pylist()
    all_decoded = pr.column("decoded_full").to_pylist()
    n_total = len(all_sids)

    if args.limit > 0 and args.limit < n_total:
        rng = np.random.default_rng(args.seed)
        pick = np.sort(rng.choice(n_total, size=args.limit, replace=False))
        sids = [all_sids[i] for i in pick]
        decoded = [all_decoded[i] for i in pick]
    else:
        sids, decoded = all_sids, all_decoded
    n = len(sids)
    print(f"generating {n} samples (batch_size={args.batch_size})")

    raw_out: list[str] = [None] * n
    t0 = time.time()
    for s in range(0, n, args.batch_size):
        batch_decoded = decoded[s:s + args.batch_size]
        prompts = [text_template.format(source_text=d) for d in batch_decoded]
        outs = generate_batch(prompts, tok, model, args.device,
                              args.max_new_tokens, args.temperature)
        for j, o in enumerate(outs):
            raw_out[s + j] = o
        done = min(s + args.batch_size, n)
        if done % (args.batch_size * 4) == 0 or done == n:
            el = time.time() - t0
            print(f"  {done}/{n}  elapsed={el:.0f}s  ({done/el:.1f}/s)", flush=True)
            if s == 0:
                print(f"  sample[0] raw[:300]: {outs[0][:300]!r}")

    cleaned = []
    parsed = []
    for r in raw_out:
        m = EXPLANATION_RE.search(r) if r else None
        if m:
            cleaned.append(m.group(1).strip()); parsed.append(True)
        else:
            cleaned.append(""); parsed.append(False)
    n_ok = sum(parsed)
    print(f"parsed: {n_ok}/{n}")

    out = pa.table({
        "sample_idx": pa.array(sids, type=pa.int64()),
        "warmstart_raw": pa.array([r or "" for r in raw_out]),
        "warmstart_cleaned": pa.array(cleaned),
        "warmstart_parsed": pa.array(parsed, type=pa.bool_()),
    })
    pq.write_table(out, args.out, compression="zstd")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
