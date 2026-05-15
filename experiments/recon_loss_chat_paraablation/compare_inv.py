"""Join the two new inverse-ablation AR outputs into the main results +
the previous ablation's output. Produces a single parquet with all
variants side by side and NMSE columns.

Variants:
    original        — full 3-paragraph AV explanation
    final_only      — paragraphs 1,2 replaced by constants, paragraph 3 kept
                      (this is the previous experiment; loaded from
                      comparison.parquet's modified_recon → nmse_modified)
    removed_final   — paragraphs 1,2 kept, paragraph 3 removed entirely
    const_final     — paragraphs 1,2 kept, paragraph 3 replaced by canned
                      "Final token 'wide'…" sentence

Output schema (per row):
    sample_idx, role, token_id, token_str, position, unpadded_pos, seq_len,
    vec_norm, activation, recon, av_parsed, explanation,
    target_logprob, target_rank, in_topk,
    recon_final_only, recon_removed_final, recon_const_final,
    nmse_original, nmse_final_only, nmse_removed_final, nmse_const_final
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _to_2d(col) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in col])


def nmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direction-NMSE (project standard, range [0,4])."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    diff = a_n - b_n
    return (diff * diff).sum(axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev-comparison", required=True,
                    help="comparison.parquet from previous ablation "
                    "(has activation + recon + modified_recon)")
    ap.add_argument("--removed-output", required=True)
    ap.add_argument("--const-output", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[cmpinv] loading prev comparison: {args.prev_comparison}")
    # Note: prev comparison may still have legacy cos_* columns from earlier
    # runs. We don't load them; cos_cols filter below also strips any that
    # might propagate via pandas operations.
    prev_columns = [
        "sample_idx", "role", "token_id", "token_str", "position",
        "unpadded_pos", "seq_len", "vec_norm",
        "activation", "recon", "av_parsed", "explanation",
        "target_logprob", "target_rank", "in_topk",
        "modified_recon", "nmse_original", "nmse_modified",
    ]
    prev = pq.read_table(args.prev_comparison,
                          columns=prev_columns).to_pandas()
    prev = prev.rename(columns={
        "modified_recon": "recon_final_only",
        "nmse_modified": "nmse_final_only",
    })
    print(f"[cmpinv]   {len(prev)} rows")

    print(f"[cmpinv] loading {args.removed_output}")
    rem = pq.read_table(args.removed_output).to_pandas().rename(
        columns={"modified_recon": "recon_removed_final"}
    )
    print(f"[cmpinv]   {len(rem)} rows")

    print(f"[cmpinv] loading {args.const_output}")
    con = pq.read_table(args.const_output).to_pandas().rename(
        columns={"modified_recon": "recon_const_final"}
    )
    print(f"[cmpinv]   {len(con)} rows")

    merged = (prev
              .merge(rem, on="sample_idx", how="inner", validate="one_to_one")
              .merge(con, on="sample_idx", how="inner", validate="one_to_one"))
    n = len(merged)
    print(f"[cmpinv] merged: {n} rows")

    act = _to_2d(merged.activation.values)
    nmse_rem = nmse(act, _to_2d(merged.recon_removed_final.values))
    nmse_con = nmse(act, _to_2d(merged.recon_const_final.values))
    merged["nmse_removed_final"] = nmse_rem
    merged["nmse_const_final"] = nmse_con

    # Drop any cos_* columns that may have leaked in from the prev comparison.
    cos_cols = [c for c in merged.columns if c.startswith("cos_")]
    if cos_cols:
        merged = merged.drop(columns=cos_cols)
        print(f"[cmpinv] dropped legacy cos columns: {cos_cols}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False),
                   args.out, compression="zstd")
    sz = Path(args.out).stat().st_size / 1e6
    print(f"[cmpinv] wrote {args.out}  ({sz:.1f} MB)")

    # Summary
    nmse_orig = merged.nmse_original.values
    nmse_fo = merged.nmse_final_only.values
    cols = [
        ("original",         nmse_orig),
        ("final_only",       nmse_fo),
        ("removed_final",    nmse_rem),
        ("const_final",      nmse_con),
    ]

    print()
    print("=== Overall ===")
    print(f"  n = {n}")
    print(f"  {'variant':<16} {'mean':>8} {'median':>8} {'p10':>8} {'p90':>8}  "
          f"{'Δmean':>8} {'%worse':>8}")
    for name, arr in cols:
        d = arr - nmse_orig
        worse = float((d > 0).mean()) * 100
        print(f"  {name:<16} {arr.mean():>8.4f} {np.median(arr):>8.4f} "
              f"{np.percentile(arr,10):>8.4f} {np.percentile(arr,90):>8.4f}  "
              f"{d.mean():>+8.4f} {worse:>7.1f}%")

    print()
    print("=== asst_content only ===")
    m = merged.role == "asst_content"
    print(f"  n = {int(m.sum())}")
    print(f"  {'variant':<16} {'mean':>8} {'median':>8} {'p10':>8} {'p90':>8}  "
          f"{'Δmean':>8} {'%worse':>8}")
    nmo_a = nmse_orig[m]
    for name, arr in cols:
        a = arr[m]
        d = a - nmo_a
        worse = float((d > 0).mean()) * 100
        print(f"  {name:<16} {a.mean():>8.4f} {np.median(a):>8.4f} "
              f"{np.percentile(a,10):>8.4f} {np.percentile(a,90):>8.4f}  "
              f"{d.mean():>+8.4f} {worse:>7.1f}%")

    print()
    print("=== By role (mean NMSE) ===")
    print(f"  {'role':<14} {'n':>6}  " +
          "  ".join(f"{c[0]:>14}" for c in cols))
    for role in sorted(merged.role.unique()):
        rm = merged.role == role
        nrow = int(rm.sum())
        vals = [arr[rm].mean() for _, arr in cols]
        print(f"  {role:<14} {nrow:>6}  " +
              "  ".join(f"{v:>14.4f}" for v in vals))


if __name__ == "__main__":
    main()
