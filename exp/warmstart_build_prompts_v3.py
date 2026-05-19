"""V3: like v2 but appends an explicit final-token hint to the prompt.

Sonnet 4.6 doesn't know Gemma's tokenization, so its <analysis> often
mis-identifies the final token. We give it the literal token verbatim by
appending after <end_text> a "Note: the final token is..." line with the
Gemma token_str JSON-escaped.

Everything else matches v2 (full chat-template prefix → tokens[:pos+1] →
decode skip_special).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

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

<begin_text>{text}<end_text>

Note: the final token in the snippet (according to the model's tokenizer, which may split words differently than you expect) is exactly: {token_json}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--convs", required=True, help="regen_wildchat_20k.jsonl")
    ap.add_argument("--tok-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("[load] tokenizer")
    tok = AutoTokenizer.from_pretrained(args.tok_dir, trust_remote_code=True)
    print(f"  chat_template: {('SET' if tok.chat_template else 'MISSING')}")

    print("[load] convs jsonl")
    convs = {}
    with open(args.convs) as f:
        for line in f:
            d = json.loads(line)
            convs[d["conv_hash"]] = d
    print(f"  {len(convs)} convs")

    print("[load] activations parquet")
    acts = pq.read_table(args.activations, columns=[
        "sample_idx", "conv_hash", "position", "unpadded_pos", "seq_len", "chars_before",
        "token_str", "n_user_turns", "n_asst_turns",
    ])

    print("[load] results parquet (for explanation_av)")
    res = pq.read_table(args.results, columns=["sample_idx", "explanation"])
    res_idx = np.asarray(res.column("sample_idx").to_pylist())
    res_map = {int(s): i for i, s in enumerate(res_idx)}

    rng = np.random.default_rng(args.seed)
    n = min(args.limit, acts.num_rows)
    pick = rng.choice(acts.num_rows, size=n, replace=False)
    pick.sort()

    # Reconstruct for each picked row
    out_records = []
    n_match_50 = 0
    n_total = 0
    failures: list[tuple[int, str]] = []
    for i in pick:
        i = int(i)
        ch = acts.column("conv_hash")[i].as_py()
        up = acts.column("unpadded_pos")[i].as_py()
        sl = acts.column("seq_len")[i].as_py()
        cb50 = acts.column("chars_before")[i].as_py()
        token_str = acts.column("token_str")[i].as_py()
        sample_idx = acts.column("sample_idx")[i].as_py()

        if ch not in convs:
            failures.append((sample_idx, f"conv_hash {ch} not in jsonl"))
            continue
        msgs = convs[ch]["messages"]

        # Render chat template to string, then tokenize. Avoids the
        # apply_chat_template(tokenize=True) return-type ambiguity (sometimes
        # list[int], sometimes BatchEncoding, sometimes Encoding).
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert isinstance(ids, list) and (not ids or isinstance(ids[0], int)), (
            f"unexpected tokenize result: type={type(ids)} first={ids[:3] if ids else None}"
        )
        # Truncate at unpadded_pos+1 (the activation position INCLUSIVE)
        if up + 1 > len(ids):
            failures.append((sample_idx, f"unpadded_pos {up} >= rendered len {len(ids)}"))
            continue
        prefix_ids = ids[:up + 1]
        decoded = tok.decode(prefix_ids, skip_special_tokens=True)
        # Sanity: decode last token (no skip_special) and compare to token_str
        last_tok_str = tok.decode([prefix_ids[-1]], skip_special_tokens=False)

        # The activation tensor has seq_len tokens (= len(ids) after potential
        # truncation to extractor.max_length). If our rendered len != seq_len,
        # truncation kicked in. Note it but still produce the prefix from OUR ids.
        len_mismatch = len(ids) != sl

        # Sanity check: chars_before is the 50 chars BEFORE the activation
        # token (exclusive). My decoded INCLUDES the activation token. So
        # the chars right before my last-token should equal cb50.
        # Use last_tok_str length to find where the current token starts.
        cut = len(decoded) - len(last_tok_str)
        match = cut >= 0 and decoded[:cut].endswith(cb50)
        n_total += 1
        n_match_50 += int(match)
        token_match = last_tok_str == token_str
        out_records.append({
            "sample_idx": sample_idx,
            "conv_hash": ch,
            "unpadded_pos": up,
            "rendered_token_len": len(ids),
            "seq_len": sl,
            "len_mismatch": len_mismatch,
            "token_str": token_str,
            "last_tok_str": last_tok_str,
            "chars_before_50": cb50,
            "decoded_full": decoded,
            "decoded_tail_50_matches_cb": match,
            "explanation_av": res.column("explanation")[res_map[sample_idx]].as_py() if sample_idx in res_map else None,
        })

    print(f"  reconstructed {len(out_records)}/{n}")
    print(f"  failures: {len(failures)}")
    if failures[:5]:
        print(f"  first failures: {failures[:5]}")
    print(f"  tail-50-before-current-token match rate: {n_match_50}/{n_total} ({100*n_match_50/max(1,n_total):.1f}%)")
    n_tok_match = sum(1 for r in out_records if r["last_tok_str"] == r["token_str"])
    print(f"  last-token == parquet token_str rate:    {n_tok_match}/{n_total} ({100*n_tok_match/max(1,n_total):.1f}%)")
    n_truncated = sum(1 for r in out_records if r["len_mismatch"])
    print(f"  rows where rendered len != seq_len (extractor truncated): {n_truncated}")

    # Print a few examples
    print()
    print("=" * 78)
    print("EXAMPLE 0:")
    print("=" * 78)
    r = out_records[0]
    print(f"sample_idx={r['sample_idx']}  unpadded_pos={r['unpadded_pos']}  rendered_len={r['rendered_token_len']}")
    print(f"chars_before_50 (from parquet): {r['chars_before_50']!r}")
    print(f"my decoded (last 200 chars):    ...{r['decoded_full'][-200:]!r}")
    print(f"tail-50 match: {r['decoded_tail_50_matches_cb']}")
    print(f"last token id decoded: {r['last_tok_str']!r}  vs parquet token_str={r['token_str']!r}")
    print(f"DECODED LEN: {len(r['decoded_full'])} chars")
    print()

    # Show length stats of the reconstructions
    decoded_lens = [len(r["decoded_full"]) for r in out_records]
    print(f"decoded prefix length stats: mean={np.mean(decoded_lens):.0f}  median={np.median(decoded_lens):.0f}  "
          f"p10={np.percentile(decoded_lens,10):.0f}  p90={np.percentile(decoded_lens,90):.0f}  max={max(decoded_lens)}")

    # Build warm-start prompts from the FULL decoded prefix + final-token hint
    prompts = [
        _DEFAULT_INSTRUCTION.format(
            text=r["decoded_full"],
            token_json=json.dumps(r["token_str"]),
        )
        for r in out_records
    ]

    out = pa.table({
        "sample_idx": pa.array([r["sample_idx"] for r in out_records], type=pa.int64()),
        "token_str": pa.array([r["token_str"] for r in out_records]),
        "conv_hash": pa.array([r["conv_hash"] for r in out_records]),
        "unpadded_pos": pa.array([r["unpadded_pos"] for r in out_records], type=pa.int32()),
        "decoded_full": pa.array([r["decoded_full"] for r in out_records]),
        "tail50_match": pa.array([r["decoded_tail_50_matches_cb"] for r in out_records], type=pa.bool_()),
        "warmstart_prompt": pa.array(prompts),
        "explanation_av": pa.array([r["explanation_av"] for r in out_records]),
    })
    pq.write_table(out, args.out, compression="zstd")
    print(f"wrote {len(out_records)} rows → {args.out}")


if __name__ == "__main__":
    main()
