"""Inverse paragraph-ablation prep.

For each av_parsed sample in results_chat_20k.parquet, emit two modified
explanations:

  variant "removed_final"
      original explanation with the final newline-split chunk dropped
      (so paragraphs 1+2 are preserved, paragraph 3 / "Final token …" is
      removed entirely).

  variant "const_final"
      paragraphs 1+2 preserved, paragraph 3 replaced with a fixed canonical
      "Final token …" sentence (chosen by user, semantically unrelated to
      the actual chat tokens — see CONST3).

Identification of "final paragraph" is consistent with the previous
ablation: it's the last newline-split chunk (`explanation.split("\n")[-1]`).
"Everything before that" is `"\n".join(lines[:-1]).rstrip()`.

Outputs two parquets:
    data/ar_input_removed_final.parquet
    data/ar_input_const_final.parquet
each with columns (sample_idx, modified_explanation).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CONST3 = (
    'Final token "wide" ends a noun phrase ("if the data has a wide...'
    'due to a wide..."), requiring a noun phrase like "range of values" '
    'or "distribution" — likely "range of values" or "unscaled range" or '
    '"of outliers" or "variance" — describing the problematic input '
    "scaling issue that causes instability in the model."
)


def before_and_after_final(expl: str) -> tuple[str, str]:
    """Return (everything-before-final-paragraph, final-paragraph).

    Consistent with previous ablation: split on '\n', last chunk is the
    final paragraph (walking back over trailing empties if needed).
    """
    if not isinstance(expl, str) or not expl:
        return "", ""
    lines = expl.split("\n")
    # Find index of last non-empty line (the actual final paragraph)
    final_i = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            final_i = i
            break
    if final_i is None:
        return "", ""
    before = "\n".join(lines[:final_i]).rstrip()
    final_para = lines[final_i]
    return before, final_para


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-parquet", required=True)
    ap.add_argument("--out-removed", required=True)
    ap.add_argument("--out-const", required=True)
    args = ap.parse_args()

    tbl = pq.read_table(
        args.results_parquet,
        columns=["sample_idx", "explanation", "av_parsed", "recon"],
    )
    df = tbl.to_pandas()
    n_total = len(df)
    mask = df.av_parsed & df.recon.apply(
        lambda v: v is not None and len(v) > 0
    )
    df = df[mask].reset_index(drop=True)
    print(f"[prep_inv] {len(df)}/{n_total} pass filter")

    befores, finals = zip(*df.explanation.apply(before_and_after_final))
    removed = list(befores)
    const = [b + "\n\n" + CONST3 for b in befores]

    n_empty_before = sum(1 for b in befores if not b)
    print(f"[prep_inv] {n_empty_before} samples have empty 'before final' "
          f"(single-paragraph originals)")

    # Show one sample
    print()
    print("[prep_inv] sample original:")
    print(df.explanation.iloc[0])
    print()
    print("[prep_inv] -> removed_final (P1+P2 only):")
    print(removed[0])
    print()
    print("[prep_inv] -> const_final (P1+P2 + canned 'wide' paragraph):")
    print(const[0])
    print()

    for path, vals in [
        (args.out_removed, removed),
        (args.out_const, const),
    ]:
        out_tbl = pa.table({
            "sample_idx": pa.array(df.sample_idx.values, pa.int64()),
            "modified_explanation": pa.array(vals, pa.string()),
        })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out_tbl, path, compression="zstd")
        sz = Path(path).stat().st_size / 1e6
        print(f"[prep_inv] wrote {path} ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
