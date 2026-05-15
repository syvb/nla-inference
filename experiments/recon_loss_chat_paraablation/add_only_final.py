"""Merge ar_output_only_final into the existing 4-variant comparison_inv
parquet, computing nmse_only_final.

Output is a new comparison parquet (or overwrite) with 5 variants.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _to_2d(col):
    return np.stack([np.asarray(v, dtype=np.float32) for v in col])


def nmse(a, b):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    diff = a_n - b_n
    return (diff * diff).sum(axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-comparison", required=True)
    ap.add_argument("--only-final-output", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[add] loading {args.in_comparison}")
    prev = pq.read_table(args.in_comparison).to_pandas()
    n = len(prev)
    print(f"[add]   {n} rows")

    print(f"[add] loading {args.only_final_output}")
    of = pq.read_table(args.only_final_output).to_pandas().rename(
        columns={"modified_recon": "recon_only_final"}
    )
    print(f"[add]   {len(of)} rows")

    merged = prev.merge(of, on="sample_idx", how="inner", validate="one_to_one")
    print(f"[add] merged: {len(merged)} rows")

    act = _to_2d(merged.activation.values)
    rec = _to_2d(merged.recon_only_final.values)
    merged["nmse_only_final"] = nmse(act, rec)

    # Drop any stray cos_* columns
    cos_cols = [c for c in merged.columns if c.startswith("cos_")]
    if cos_cols:
        merged = merged.drop(columns=cos_cols)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False),
                   args.out, compression="zstd")
    sz = Path(args.out).stat().st_size / 1e6
    print(f"[add] wrote {args.out}  ({sz:.1f} MB)")

    # Summary
    variants = [
        ("nmse_original",      "original"),
        ("nmse_final_only",    "final-¶ only (P1,P2 replaced)"),
        ("nmse_only_final",    "only_final (just P3, no prelude)"),
        ("nmse_removed_final", "removed_final (P3 dropped)"),
        ("nmse_const_final",   "const_final (P3 = canned 'wide')"),
    ]
    nmo = merged.nmse_original.values
    print()
    print("=== Overall ===")
    print(f"  {'variant':<40} {'mean':>8} {'median':>8} {'Δmean':>9} {'%worse':>8}")
    for col, lbl in variants:
        v = merged[col].values
        d = v - nmo
        print(f"  {lbl:<40} {v.mean():>8.4f} {np.median(v):>8.4f} "
              f"{d.mean():>+9.4f} {100*(d>0).mean():>7.1f}%")

    m = merged.role == "asst_content"
    nmo_a = nmo[m]
    print()
    print(f"=== asst_content only (n={int(m.sum())}) ===")
    print(f"  {'variant':<40} {'mean':>8} {'median':>8} {'Δmean':>9} {'%worse':>8}")
    for col, lbl in variants:
        v = merged[col].values[m]
        d = v - nmo_a
        print(f"  {lbl:<40} {v.mean():>8.4f} {np.median(v):>8.4f} "
              f"{d.mean():>+9.4f} {100*(d>0).mean():>7.1f}%")


if __name__ == "__main__":
    main()
