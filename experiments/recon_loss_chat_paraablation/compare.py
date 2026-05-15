"""Join modified_recon into the main chat results and compute side-by-side NMSE.

Output: a parquet with per-sample (sample_idx, role, token_str,
nmse_original, nmse_modified, delta_nmse) plus the activations and both
recons for further analysis.

NMSE is reported as the project's standard normalized direction-MSE
(range [0, 4]; 0 = perfect, 2 = orthogonal). Summary stats and tables
report NMSE only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _to_2d(col):
    return np.stack([np.asarray(v, dtype=np.float32) for v in col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-parquet", required=True,
                    help="results_chat_20k.parquet (has activation + original recon)")
    ap.add_argument("--ar-output", required=True,
                    help="ar_output.parquet (has modified_recon)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[cmp] loading results: {args.results_parquet}")
    res_cols = ["sample_idx", "role", "token_id", "token_str", "position",
                "unpadded_pos", "seq_len", "vec_norm", "activation", "recon",
                "explanation", "av_parsed",
                "target_logprob", "target_rank", "in_topk"]
    res = pq.read_table(args.results_parquet, columns=res_cols).to_pandas()
    print(f"[cmp]   {len(res)} rows")

    print(f"[cmp] loading ar output: {args.ar_output}")
    ar = pq.read_table(args.ar_output).to_pandas()
    print(f"[cmp]   {len(ar)} rows")

    merged = res.merge(ar, on="sample_idx", how="inner", validate="one_to_one")
    n = len(merged)
    print(f"[cmp] merged: {n} rows")

    act = _to_2d(merged.activation.values)
    rec_orig = _to_2d(merged.recon.values)
    rec_mod = _to_2d(merged.modified_recon.values)

    def nmse(a, b):
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        diff = a_n - b_n
        return (diff * diff).sum(axis=1).astype(np.float32)

    nmse_o = nmse(act, rec_orig)
    nmse_m = nmse(act, rec_mod)
    delta = nmse_m - nmse_o

    merged["nmse_original"] = nmse_o
    merged["nmse_modified"] = nmse_m
    merged["delta_nmse"] = delta

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False),
                   args.out, compression="zstd")
    sz = Path(args.out).stat().st_size / 1e6
    print(f"[cmp] wrote {args.out} ({sz:.1f} MB)")

    print()
    print("=== Overall ===")
    print(f"  n = {n}")
    print(f"  NMSE original:  mean={nmse_o.mean():.4f}  median={np.median(nmse_o):.4f}  "
          f"p10={np.percentile(nmse_o,10):.4f}  p90={np.percentile(nmse_o,90):.4f}")
    print(f"  NMSE modified:  mean={nmse_m.mean():.4f}  median={np.median(nmse_m):.4f}  "
          f"p10={np.percentile(nmse_m,10):.4f}  p90={np.percentile(nmse_m,90):.4f}")
    print(f"  delta NMSE:     mean={delta.mean():+.4f}  median={np.median(delta):+.4f}  "
          f"p10={np.percentile(delta,10):+.4f}  p90={np.percentile(delta,90):+.4f}")
    print(f"  worse: {(delta>0).sum()} ({100*(delta>0).mean():.1f}%)   "
          f"better: {(delta<0).sum()} ({100*(delta<0).mean():.1f}%)")

    print()
    print("=== By role ===")
    print(f"  {'role':<14} {'n':>6}  {'nmse_o':>7} {'nmse_m':>7} {'delta':>7}")
    for role in sorted(merged.role.unique()):
        m = merged.role == role
        n_r = int(m.sum())
        no = float(nmse_o[m].mean())
        nm = float(nmse_m[m].mean())
        print(f"  {role:<14} {n_r:>6}  {no:>7.4f} {nm:>7.4f} {nm-no:>+7.4f}")


if __name__ == "__main__":
    main()
