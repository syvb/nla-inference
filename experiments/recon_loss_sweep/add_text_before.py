"""Add a `text_before` column to existing results / activations parquets.

For each row we have the tokenized `position` and the `text_preview`
(first 200 chars of the original document). The original document text was
NOT stored, but the streaming sampler is deterministic given seed + mode, so
we can re-stream the dataset, match each row by text_preview, re-tokenize
the matched text with the same max_len truncation, and decode `[:position]`
to recover the exact text the model saw before the token.

Output: writes a NEW parquet next to each input with `_with_text_before.parquet`
suffix that has every original column plus a new `text_before` column.

Tokenizer comes from the base-model HF repo (only the tokenizer files need
to be downloaded, ~5-15 MB, not the multi-GB model weights).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# Reuse the samplers from the main script so the order matches exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from recon_loss_sweep import diverse_shard_iter, streaming_default_iter


def replay(parquet_path: Path, *, tokenizer_repo: str, dataset: str,
           sampling: str, seed: int, max_len: int = 512,
           out_path: Path | None = None) -> Path:
    from transformers import AutoTokenizer

    print(f"\n[replay] {parquet_path}")
    print(f"  tokenizer = {tokenizer_repo}")
    print(f"  sampling  = {sampling}  seed = {seed}  max_len = {max_len}")

    tok = AutoTokenizer.from_pretrained(tokenizer_repo, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tbl = pq.read_table(parquet_path)
    df = tbl.to_pandas()
    n = len(df)
    print(f"  rows = {n}")

    # text_preview → first index (handle dups: pick the smallest unmatched idx
    # so a single duplicated preview doesn't collide.)
    from collections import defaultdict
    preview_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(df["text_preview"]):
        preview_to_idxs[p].append(i)

    text_before = [None] * n
    matched = 0
    seen_texts = 0

    # For replay, the SET of texts is determined by the sampler+seed+shard
    # selection. The exact order depends on shuffle buffer, but order doesn't
    # matter for text_preview matching — just use a small buffer to save RAM.
    if sampling == "diverse_shards":
        src = diverse_shard_iter(dataset, seed, buffer_size=2000)
    elif sampling == "streaming_default":
        src = streaming_default_iter(dataset, "train", seed, buffer_size=2000)
    else:
        raise ValueError(sampling)

    # We can stop early once everything is matched.
    target = sum(1 for v in preview_to_idxs.values() if v)

    for text in src:
        seen_texts += 1
        prev = text[:200]
        idxs = preview_to_idxs.get(prev)
        if not idxs:
            continue
        # Take the first unmatched idx for this preview
        idx = idxs.pop(0)
        if not idxs:
            del preview_to_idxs[prev]

        # Re-tokenize with same truncation; text_before = decode(ids[:position]).
        ids = tok(text, truncation=True, max_length=max_len,
                  add_special_tokens=True)["input_ids"]
        pos = int(df.at[idx, "position"])
        if pos > 0 and pos <= len(ids):
            text_before[idx] = tok.decode(ids[:pos], skip_special_tokens=False)
        else:
            text_before[idx] = ""
        matched += 1

        if matched % 2000 == 0:
            print(f"  ...matched {matched}/{n}  seen={seen_texts}", flush=True)

        if not preview_to_idxs:
            break

    print(f"[replay] matched {matched}/{n}  (saw {seen_texts} streamed texts)")
    if matched < n:
        print(f"[replay] WARNING: {n - matched} rows unmatched; preview collisions or "
              f"sampler nondeterminism. Their text_before is None.")

    df["text_before"] = text_before
    new_tbl = pa.Table.from_pandas(df, preserve_index=False)
    out_path = out_path or parquet_path.with_name(
        parquet_path.stem + "_with_text_before.parquet")
    pq.write_table(new_tbl, out_path, compression="zstd")
    print(f"[replay] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="Process all 4 known parquets in data/.")
    ap.add_argument("--parquet")
    ap.add_argument("--tokenizer-repo")
    ap.add_argument("--sampling", default="diverse_shards")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(__file__).resolve().parent / "data"

    if args.all:
        # (parquet_filename, tokenizer_repo, sampling)
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
        for fname, tok, samp in jobs:
            p = data / fname
            if not p.exists():
                print(f"[skip] {p} missing"); continue
            replay(p, tokenizer_repo=tok,
                   dataset="common-pile/comma_v0.1_training_dataset",
                   sampling=samp, seed=args.seed)
    else:
        if not args.parquet or not args.tokenizer_repo:
            ap.error("provide --parquet + --tokenizer-repo, or --all")
        replay(Path(args.parquet),
               tokenizer_repo=args.tokenizer_repo,
               dataset="common-pile/comma_v0.1_training_dataset",
               sampling=args.sampling, seed=args.seed)


if __name__ == "__main__":
    main()
