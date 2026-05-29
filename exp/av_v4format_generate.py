"""Run the AV model with the v4 Sonnet prompt scaffold (few-shot, warm-start).

Unlike av_on_text_generate.py (which used the AV's native <concept> prompt,
zero-shot), this builds the SAME multi-turn few-shot prompt that worked for
Sonnet in the v4 experiment:
  [user: stage2 warm-start instruction w/ example text]
  [assistant: <analysis>...example AV explanation...</analysis>]
  ... (N shots) ...
  [user: stage2 warm-start instruction w/ the eval text]
and feeds it to the AV model (no injection). Tests whether few-shot + the
better scaffold helps the AV as a text explainer.

Few-shot examples are drawn from chat rows OUTSIDE the eval subset.
Output schema matches the warmstart parquets.
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

# Accept either tag — few-shot demos use <analysis>, but the AV is RL-trained
# to emit <explanation>, so it may revert.
TAG_RE = re.compile(r"<(analysis|explanation)>\s*(.*?)\s*</\1>", re.DOTALL)


@torch.inference_mode()
def generate_batch(message_lists, tok, model, device, max_new_tokens, temperature):
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    texts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in message_lists
    ]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=8192, add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    gen = model.generate(
        input_ids=ids, attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0), temperature=temperature,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    new = gen[:, ids.shape[1]:]
    return [tok.decode(row, skip_special_tokens=True) for row in new]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-ckpt", required=True)
    ap.add_argument("--prompts", required=True,
                    help="prompts_chat_20k.parquet (warmstart_prompt + explanation_av)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--n-shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot-seed", type=int, default=42)
    args = ap.parse_args()

    pr = pq.read_table(args.prompts)
    sids = pr.column("sample_idx").to_pylist()
    wprompt = pr.column("warmstart_prompt").to_pylist()
    av_expl = pr.column("explanation_av").to_pylist()
    n_total = len(sids)

    # Eval subset (same seed-0 5k as the §5b run).
    rng = np.random.default_rng(args.seed)
    if args.limit > 0 and args.limit < n_total:
        eval_pick = np.sort(rng.choice(n_total, size=args.limit, replace=False))
    else:
        eval_pick = np.arange(n_total)
    eval_set = set(int(i) for i in eval_pick)

    # Few-shot examples: rows OUTSIDE the eval subset with a non-empty AV expl.
    fs_rng = np.random.default_rng(args.fewshot_seed)
    candidates = [i for i in range(n_total)
                  if i not in eval_set and av_expl[i] and wprompt[i]]
    fs_idx = fs_rng.choice(candidates, size=args.n_shots, replace=False)
    shots = []
    for i in fs_idx:
        i = int(i)
        shots.append((wprompt[i], f"<analysis>\n{av_expl[i]}\n</analysis>"))
        print(f"  shot sample_idx={sids[i]}  prompt_chars={len(wprompt[i])}  expl_chars={len(av_expl[i])}")

    def build_messages(eval_prompt):
        msgs = []
        for up, resp in shots:
            msgs.append({"role": "user", "content": up})
            msgs.append({"role": "assistant", "content": resp})
        msgs.append({"role": "user", "content": eval_prompt})
        return msgs

    print(f"[load] {args.av_ckpt}")
    tok = AutoTokenizer.from_pretrained(args.av_ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.av_ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()

    eval_sids = [sids[int(i)] for i in eval_pick]
    eval_prompts = [wprompt[int(i)] for i in eval_pick]
    n = len(eval_sids)
    print(f"generating {n} samples ({args.n_shots}-shot, batch_size={args.batch_size})")

    raw_out = [None] * n
    t0 = time.time()
    for s in range(0, n, args.batch_size):
        batch = [build_messages(p) for p in eval_prompts[s:s + args.batch_size]]
        outs = generate_batch(batch, tok, model, args.device,
                              args.max_new_tokens, args.temperature)
        for j, o in enumerate(outs):
            raw_out[s + j] = o
        done = min(s + args.batch_size, n)
        if done % (args.batch_size * 4) == 0 or done == n:
            el = time.time() - t0
            print(f"  {done}/{n}  elapsed={el:.0f}s  ({done/el:.2f}/s)", flush=True)
            if s == 0:
                print(f"  sample[0] raw[:300]: {outs[0][:300]!r}")

    cleaned, parsed = [], []
    for r in raw_out:
        m = TAG_RE.search(r) if r else None
        if m:
            cleaned.append(m.group(2).strip()); parsed.append(True)
        else:
            cleaned.append(""); parsed.append(False)
    print(f"parsed: {sum(parsed)}/{n}")

    out = pa.table({
        "sample_idx": pa.array(eval_sids, type=pa.int64()),
        "warmstart_raw": pa.array([r or "" for r in raw_out]),
        "warmstart_cleaned": pa.array(cleaned),
        "warmstart_parsed": pa.array(parsed, type=pa.bool_()),
    })
    pq.write_table(out, args.out, compression="zstd")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
