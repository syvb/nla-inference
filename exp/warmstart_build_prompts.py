"""Build warm-start prompts (stage2 _DEFAULT_INSTRUCTION) for 500 sample_idx.

Joins activations_chat_20k (has chars_before) with results_chat_20k (has AV
explanation + recon + activation) on sample_idx. Writes prompts.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Verbatim from natural_language_autoencoders/nla/datagen/stage2_api_explain.py
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acts = pq.read_table(args.activations)
    res = pq.read_table(args.results)
    print(f"activations: {acts.num_rows} rows, cols={acts.column_names}")
    print(f"results:     {res.num_rows} rows, cols={res.column_names}")

    # Join on sample_idx — both should be identically ordered/indexed
    res_idx = np.asarray(res.column("sample_idx").to_pylist())
    act_idx = np.asarray(acts.column("sample_idx").to_pylist())
    assert (res_idx == act_idx).all(), "sample_idx order differs"

    rng = np.random.default_rng(args.seed)
    n = min(args.limit, res.num_rows)
    pick = rng.choice(res.num_rows, size=n, replace=False)
    pick.sort()

    chars_before = [acts.column("chars_before")[int(i)].as_py() for i in pick]
    token_str = [acts.column("token_str")[int(i)].as_py() for i in pick]
    sample_idx = [int(res_idx[i]) for i in pick]
    explanation_av = [res.column("explanation")[int(i)].as_py() for i in pick]

    prompts = [_DEFAULT_INSTRUCTION.format(text=t) for t in chars_before]

    # Print one example for sanity
    print("=" * 78)
    print("EXAMPLE PROMPT (first sample):")
    print("-" * 78)
    print(prompts[0][:2000])
    print("..." if len(prompts[0]) > 2000 else "")
    print("=" * 78)
    print("EXISTING AV EXPLANATION (first sample):")
    print("-" * 78)
    print(explanation_av[0])
    print("=" * 78)

    out = pa.table({
        "sample_idx": pa.array(sample_idx, type=pa.int64()),
        "token_str": pa.array(token_str),
        "chars_before": pa.array(chars_before),
        "warmstart_prompt": pa.array(prompts),
        "explanation_av": pa.array(explanation_av),
    })
    pq.write_table(out, args.out, compression="zstd")
    print(f"wrote {n} rows → {args.out}")


if __name__ == "__main__":
    main()
