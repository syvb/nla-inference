"""Build the AR-input parquet for the paragraph-ablation experiment.

For each sample in results_chat_20k.parquet that has av_parsed=True and a
non-null recon, replace the original AV explanation with:

    CONST1 + "\n\n" + CONST2 + "\n\n" + final_paragraph

where `final_paragraph` is the last `\n`-split line of the original
explanation (or the whole string if there are no newlines).

The two constants are fixed canonical "structure" / "sentence setup"
paragraphs picked by the user — they replace the per-sample first two
paragraphs of the original 3-paragraph AV output.

Output a small parquet with just (sample_idx, modified_explanation) to ship
to the remote AR host. Everything else (activation, original recon, etc.)
stays here for the post-hoc NMSE join.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CONST1 = (
    "Structured ML/data science explanation format: structured advice "
    "with code blocks and conceptual framing establishes a technical "
    "troubleshooting guide."
)
CONST2 = (
    "The sentence \"If your data has a wide\" sets up a problem statement "
    "about batch normalization instability, specifically the issue of "
    "feature scaling or a large input range."
)


def build_modified(expl: str) -> str:
    if not isinstance(expl, str) or not expl:
        final_para = ""
    else:
        lines = expl.split("\n")
        # Last non-empty line — but per user spec we split on any \n and take
        # the last element. If that's empty (trailing newline), back up.
        final_para = lines[-1]
        if not final_para.strip():
            # Walk backward until we find non-empty content
            for l in reversed(lines):
                if l.strip():
                    final_para = l
                    break
    return f"{CONST1}\n\n{CONST2}\n\n{final_para}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-parquet", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tbl = pq.read_table(
        args.results_parquet,
        columns=["sample_idx", "explanation", "av_parsed", "recon"],
    )
    df = tbl.to_pandas()
    n_total = len(df)

    # Filter
    mask = df.av_parsed & df.recon.apply(
        lambda v: v is not None and len(v) > 0
    )
    df = df[mask].reset_index(drop=True)
    n_keep = len(df)
    print(f"[prep] {n_keep}/{n_total} pass filter (av_parsed & recon non-null)")

    # Build modified explanations
    df["modified_explanation"] = df.explanation.apply(build_modified)

    # Sanity: count single-paragraph originals
    n_single = (df.explanation.str.count("\n") == 0).sum()
    print(f"[prep] {n_single} originals are single-line (whole used as final_para)")

    # Show a sample
    print("\n[prep] sample modified explanation:")
    print(df.modified_explanation.iloc[0])
    print()
    print("[prep] original final_para of that sample (sanity):")
    print(df.explanation.iloc[0].split("\n")[-1])
    print()

    out_tbl = pa.table({
        "sample_idx": pa.array(df.sample_idx.values, pa.int64()),
        "modified_explanation": pa.array(df.modified_explanation.values, pa.string()),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_tbl, args.out, compression="zstd")
    sz = Path(args.out).stat().st_size / 1e6
    print(f"[prep] wrote {args.out}  ({sz:.2f} MB, {len(df)} rows)")


if __name__ == "__main__":
    main()
