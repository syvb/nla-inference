"""Augment v7 explanations by appending the original input text.

For each row in the v7 Sonnet parquet, build a new "warmstart_cleaned"
column that is the v7 explanation followed by the original input prefix
(decoded_full from the prompts parquet). The AR critic then sees both the
analysis-style explanation AND the raw text it's analyzing.

Output parquet is otherwise identical in schema so the existing
warmstart_run_ar_multigpu.py can consume it.
"""
from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonnet", required=True, help="v7 sonnet parquet")
    ap.add_argument("--prompts", required=True, help="prompts parquet with decoded_full")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s = pq.read_table(args.sonnet)
    p = pq.read_table(args.prompts)
    p_idx = {p.column("sample_idx")[i].as_py(): i for i in range(p.num_rows)}

    sids = s.column("sample_idx").to_pylist()
    cleaned = s.column("warmstart_cleaned").to_pylist()
    raw = s.column("warmstart_raw").to_pylist()
    parsed = s.column("warmstart_parsed").to_pylist()

    aug = []
    for i in range(len(sids)):
        sid = sids[i]
        if sid not in p_idx:
            aug.append(cleaned[i])
            continue
        original = p.column("decoded_full")[p_idx[sid]].as_py()
        if parsed[i] and cleaned[i]:
            aug.append(f"{cleaned[i]}\n\n{original}")
        else:
            aug.append(cleaned[i])

    out = pa.table({
        "sample_idx": pa.array(sids, type=pa.int64()),
        "warmstart_raw": pa.array(raw),
        "warmstart_cleaned": pa.array(aug),
        "warmstart_parsed": pa.array(parsed, type=pa.bool_()),
    })
    pq.write_table(out, args.out, compression="zstd")
    print(f"wrote {args.out}")
    # Sample
    print()
    print("--- EXAMPLE (first parsed row) ---")
    for i in range(len(sids)):
        if parsed[i]:
            print(f"sample_idx={sids[i]}")
            print(f"augmented length: {len(aug[i])} chars (was {len(cleaned[i])})")
            print(f"first 300:")
            print(aug[i][:300])
            print("...")
            print(f"last 300:")
            print(aug[i][-300:])
            break


if __name__ == "__main__":
    main()
