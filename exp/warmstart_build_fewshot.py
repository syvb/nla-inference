"""Build a 4-example few-shot prefix for the warm-start prompt.

Picks 4 rows from the chat 20k dataset that are NOT in the eval set, builds
(warm_start_prompt, av_explanation) pairs, and saves them as JSON.

Few-shot prompt format follows v2 (full chat-template prefix → tokens[:pos+1]
→ decode skip_special). The AV's `explanation` column is the "demonstration"
answer — it's what the trained AV produces given activation access, which
represents the target structure/style we want Sonnet to mimic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# Same instruction as v2 (no token hint — that's a separate axis).
_DEFAULT_INSTRUCTION = """A language model needs to predict what text comes next after a snippet which will be presented to you shortly. Identify the 2-3 most important features it would use for this prediction.
Focus on what the language model must be "thinking about" at the point where the provided text ends. You should not need to reference the fact that the text is truncated/incomplete/a prefix: the language model is causal, so only sees the prefix to what it predicts and this is implicit.
Order features by what is most important for predicting the next tokens. Each feature should consist of a concise ~10-20 word description. Feel free to include specific textual examples inline.

Feature types to consider (as inspiration, not a rigid checklist):
- Syntactic/structural constraints: "unclosed parenthesis requires matching close"
- Immediate semantic expectations: "list promised three items but only two given"
- Stylistic/register patterns: "formal academic tone maintained throughout"
- Narrative/argumentative momentum: "thesis stated, supporting evidence now expected"
- Domain/genre signals: "medical case history following SOAP format"
- Repetition/continuation patterns: "same phrase structure repeating with variations"

The final feature must describe the very end of the presented sequence: its role, what it's part of, and immediate constraints on what follows.

Format — IMPORTANT: keep to ~80-100 words total and ALWAYS close the tag:
<analysis>
[first feature — include specific examples when relevant]
[second feature]
[final feature: the last token, its role, immediate constraints]
</analysis>

Text to analyze:

<begin_text>{text}<end_text>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--convs", required=True)
    ap.add_argument("--tok-dir", required=True)
    ap.add_argument("--eval-prompts", required=True,
                    help="prompts_500_v2.parquet — used to know which sample_idx to AVOID")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--n-shots", type=int, default=4)
    ap.add_argument("--fewshot-seed", type=int, default=42,
                    help="distinct from eval seed (0)")
    args = ap.parse_args()

    print("[load] tokenizer")
    tok = AutoTokenizer.from_pretrained(args.tok_dir, trust_remote_code=True)

    print("[load] convs jsonl")
    convs = {}
    with open(args.convs) as f:
        for line in f:
            d = json.loads(line)
            convs[d["conv_hash"]] = d
    print(f"  {len(convs)} convs")

    print("[load] eval prompts (sample_idx to exclude)")
    eval_tbl = pq.read_table(args.eval_prompts, columns=["sample_idx"])
    eval_sids = set(eval_tbl.column("sample_idx").to_pylist())
    print(f"  excluding {len(eval_sids)} eval sample_idx")

    print("[load] activations + results")
    acts = pq.read_table(args.activations, columns=[
        "sample_idx", "conv_hash", "unpadded_pos", "token_str",
    ])
    res = pq.read_table(args.results, columns=["sample_idx", "explanation", "av_parsed"])

    sids = np.asarray(acts.column("sample_idx").to_pylist())
    # Build index for results
    res_idx = np.asarray(res.column("sample_idx").to_pylist())
    res_map = {int(s): i for i, s in enumerate(res_idx)}

    # Eligible pool: not in eval set, AV parsed, conv_hash present in jsonl
    candidates = []
    for i in range(acts.num_rows):
        sid = int(sids[i])
        if sid in eval_sids:
            continue
        if sid not in res_map:
            continue
        if not res.column("av_parsed")[res_map[sid]].as_py():
            continue
        ch = acts.column("conv_hash")[i].as_py()
        if ch not in convs:
            continue
        candidates.append(i)
    print(f"  eligible pool: {len(candidates)} rows")

    rng = np.random.default_rng(args.fewshot_seed)
    pick = rng.choice(candidates, size=args.n_shots, replace=False)

    shots = []
    for i in pick:
        i = int(i)
        sid = int(sids[i])
        ch = acts.column("conv_hash")[i].as_py()
        up = acts.column("unpadded_pos")[i].as_py()
        token_str = acts.column("token_str")[i].as_py()
        msgs = convs[ch]["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if up + 1 > len(ids):
            print(f"  WARN: sample {sid} pos {up} >= len(ids) {len(ids)} — skipping")
            continue
        decoded = tok.decode(ids[:up + 1], skip_special_tokens=True)
        prompt = _DEFAULT_INSTRUCTION.format(text=decoded)
        # AV explanation is the demonstration answer. Wrap in <analysis>...</analysis>
        # to match what we'd extract from a regular model response.
        av_expl = res.column("explanation")[res_map[sid]].as_py()
        response = f"<analysis>\n{av_expl}\n</analysis>"
        shots.append({
            "sample_idx": sid,
            "conv_hash": ch,
            "unpadded_pos": up,
            "token_str": token_str,
            "decoded_len_chars": len(decoded),
            "prompt": prompt,
            "response": response,
        })
        print(f"  shot sample_idx={sid}  decoded_len={len(decoded):4d} chars  "
              f"av_expl_len={len(av_expl):4d} chars  token={token_str!r}")

    Path(args.out).write_text(json.dumps(shots, indent=2))
    print(f"wrote {len(shots)} shots → {args.out}")


if __name__ == "__main__":
    main()
