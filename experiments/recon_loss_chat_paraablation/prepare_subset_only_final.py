"""Prep for the only-final variant: explanation = P3 alone (no prelude).

Same final-paragraph extraction as prepare_subset_inv (last non-empty
newline-split chunk), but emit *just* that chunk with no surrounding
text or paragraphs. Compare to the previous final-only variant, which
was CONST1 + CONST2 + P3.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def final_paragraph(expl: str) -> str:
    if not isinstance(expl, str) or not expl:
        return ""
    for line in reversed(expl.split("\n")):
        if line.strip():
            return line
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-parquet", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pq.read_table(args.results_parquet,
                        columns=["sample_idx", "explanation",
                                  "av_parsed", "recon"]).to_pandas()
    mask = df.av_parsed & df.recon.apply(
        lambda v: v is not None and len(v) > 0
    )
    df = df[mask].reset_index(drop=True)
    print(f"[only_final] {len(df)} rows pass filter")

    finals = df.explanation.apply(final_paragraph)
    n_empty = int((finals.str.strip() == "").sum())
    print(f"[only_final] {n_empty} samples with empty final paragraph")

    print()
    print("[only_final] sample original:")
    print(df.explanation.iloc[0])
    print()
    print("[only_final] -> only_final:")
    print(finals.iloc[0])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "sample_idx": pa.array(df.sample_idx.values, pa.int64()),
            "modified_explanation": pa.array(finals.values, pa.string()),
        }),
        args.out, compression="zstd",
    )
    sz = Path(args.out).stat().st_size / 1e6
    print(f"\n[only_final] wrote {args.out} ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
