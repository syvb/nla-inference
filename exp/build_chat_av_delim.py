"""Build chat av_delim parquet (input + AV explanation) for all 20k rows.

Output schema mirrors the augmented-sonnet parquets so warmstart_run_ar_multi.py
can consume it. Drops rows where AV failed to parse OR prefix reconstruction failed.
"""
from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="prompts_chat_20k.parquet")
    ap.add_argument("--results", required=True, help="results_chat_20k.parquet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = pq.read_table(args.prompts, columns=["sample_idx", "decoded_full", "explanation_av"])
    r = pq.read_table(args.results, columns=["sample_idx", "explanation", "av_parsed"])
    r_idx = {r.column("sample_idx")[i].as_py(): i for i in range(r.num_rows)}

    sids = p.column("sample_idx").to_pylist()
    decoded = p.column("decoded_full").to_pylist()

    aug = []
    parsed = []
    raw = []
    for i, sid in enumerate(sids):
        if sid not in r_idx:
            aug.append(""); parsed.append(False); raw.append("")
            continue
        j = r_idx[sid]
        av_ok = r.column("av_parsed")[j].as_py()
        if not av_ok:
            aug.append(""); parsed.append(False); raw.append("")
            continue
        av_expl = r.column("explanation")[j].as_py()
        if not av_expl or not decoded[i]:
            aug.append(""); parsed.append(False); raw.append("")
            continue
        aug.append(f"Original text:\n{decoded[i]}\n\nAnalysis:\n{av_expl}")
        parsed.append(True)
        raw.append("")

    n_ok = sum(parsed)
    print(f"av_delim valid: {n_ok}/{len(sids)}")
    out = pa.table({
        "sample_idx": pa.array(sids, type=pa.int64()),
        "warmstart_raw": pa.array(raw),
        "warmstart_cleaned": pa.array(aug),
        "warmstart_parsed": pa.array(parsed, type=pa.bool_()),
    })
    pq.write_table(out, args.out, compression="zstd")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
