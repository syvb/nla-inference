"""Add `chars_before` (50 chars before the token) and `chars_after`
(50 chars after the token) columns to existing results parquets.

For each row we have the tokenized `position` and the `text_preview`
(first 200 chars of the original document). The original document text was
not stored, but the streaming sampler is deterministic given seed + mode, so
we re-stream the dataset, match each row by text_preview, re-tokenize the
matched text with the same max_len truncation, and slice the decoded
context around `position`.

Output: writes a NEW parquet next to each input with `_with_context.parquet`
suffix that has every original column plus `chars_before` and `chars_after`.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from recon_loss_sweep import diverse_shard_iter, streaming_default_iter


WINDOW = 50  # chars before / after


def replay(parquet_path: Path, *, tokenizer_repo: str, dataset: str,
           sampling: str, seed: int, max_len: int = 512,
           out_path: Path | None = None) -> Path:
    from transformers import AutoTokenizer

    print(f"\n[replay] {parquet_path}")
    print(f"  tokenizer = {tokenizer_repo}  sampling = {sampling}  seed = {seed}")

    tok = AutoTokenizer.from_pretrained(tokenizer_repo, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"  padding_side = {tok.padding_side}")

    df = pq.read_table(parquet_path).to_pandas()
    n = len(df)
    print(f"  rows = {n}")

    preview_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(df["text_preview"]):
        preview_to_idxs[p].append(i)

    chars_before = [None] * n
    chars_after = [None] * n
    matched = 0
    seen = 0

    if sampling == "diverse_shards":
        src = diverse_shard_iter(dataset, seed, buffer_size=2000)
    elif sampling == "streaming_default":
        src = streaming_default_iter(dataset, "train", seed, buffer_size=2000)
    else:
        raise ValueError(sampling)

    for text in src:
        seen += 1
        prev = text[:200]
        idxs = preview_to_idxs.get(prev)
        if not idxs:
            continue
        idx = idxs.pop(0)
        if not idxs:
            del preview_to_idxs[prev]

        ids = tok(text, truncation=True, max_length=max_len,
                  add_special_tokens=True)["input_ids"]
        # The recorded `position` is in the PADDED batch tensor. The original
        # extract batch-tokenized with padding=True, which left-pads for Gemma
        # and right-pads for Qwen. So:
        #   left-pad:  unpadded_pos = position - (max_len - seq_len)
        #   right-pad: unpadded_pos = position  (no offset)
        pos = int(df.at[idx, "position"])
        seq_len = int(df.at[idx, "seq_len"])
        if tok.padding_side == "left":
            pad_offset = max_len - seq_len
            unpadded_pos = pos - pad_offset
        else:
            unpadded_pos = pos
        if 0 <= unpadded_pos < len(ids):
            before = tok.decode(ids[:unpadded_pos], skip_special_tokens=False)
            after  = tok.decode(ids[unpadded_pos+1:], skip_special_tokens=False)
            chars_before[idx] = before[-WINDOW:] if len(before) > WINDOW else before
            chars_after[idx]  = after[:WINDOW]
        else:
            chars_before[idx] = ""
            chars_after[idx] = ""
        matched += 1
        if matched % 2000 == 0:
            print(f"  ...matched {matched}/{n}  seen={seen}", flush=True)
        if not preview_to_idxs:
            break

    print(f"[replay] matched {matched}/{n}  (saw {seen} streamed texts)")
    if matched < n:
        unmatched = n - matched
        print(f"[replay] WARNING: {unmatched} rows unmatched")

    df["chars_before"] = chars_before
    df["chars_after"] = chars_after

    out_path = out_path or parquet_path.with_name(
        parquet_path.stem + "_with_context.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   out_path, compression="zstd")
    print(f"[replay] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--parquet")
    ap.add_argument("--tokenizer-repo")
    ap.add_argument("--sampling", default="diverse_shards")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(__file__).resolve().parent / "data"
    DATASET = "common-pile/comma_v0.1_training_dataset"

    if args.all:
        jobs = [
            ("results_20000.parquet",
             "google/gemma-3-12b-it", "streaming_default"),
            ("results_gemma12_diverse_shards_seed0_20000.parquet",
             "google/gemma-3-12b-it", "diverse_shards"),
            ("results_qwen7_20000.parquet",
             "Qwen/Qwen2.5-7B-Instruct", "streaming_default"),
            ("results_qwen7_diverse_shards_seed0_20000.parquet",
             "Qwen/Qwen2.5-7B-Instruct", "diverse_shards"),
        ]
        for fname, t, samp in jobs:
            p = data / fname
            if not p.exists():
                print(f"[skip] {p} missing"); continue
            replay(p, tokenizer_repo=t, dataset=DATASET,
                   sampling=samp, seed=args.seed)
    else:
        if not args.parquet or not args.tokenizer_repo:
            ap.error("provide --parquet + --tokenizer-repo, or --all")
        replay(Path(args.parquet), tokenizer_repo=args.tokenizer_repo,
               dataset=DATASET, sampling=args.sampling, seed=args.seed)


if __name__ == "__main__":
    main()
