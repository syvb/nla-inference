"""Plot the NMSE distribution shift on assistant-content tokens only.

Emits two separate PNGs:
  --out-dist:  overlaid histograms of NMSE_original vs NMSE_<variant>
  --out-delta: histogram of per-sample Δ NMSE = NMSE_<variant> − NMSE_original
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-parquet", required=True)
    ap.add_argument("--variant-col", default="nmse_modified",
                    help="name of the variant's NMSE column to compare "
                         "against nmse_original")
    ap.add_argument("--variant-label", default="modified (2 constants + final ¶)",
                    help="legend/title label for the variant")
    ap.add_argument("--out-dist", required=True)
    ap.add_argument("--out-delta", required=True)
    args = ap.parse_args()

    print(f"[plot] loading {args.comparison_parquet}")
    df = pq.read_table(args.comparison_parquet,
                        columns=["role", "nmse_original",
                                 args.variant_col]).to_pandas()
    df = df[df.role == "asst_content"].reset_index(drop=True)
    n = len(df)
    print(f"[plot] n = {n} asst_content rows  variant={args.variant_col}")

    nmse_o = df.nmse_original.values
    nmse_m = df[args.variant_col].values
    delta = nmse_m - nmse_o

    print(f"  NMSE orig  mean={nmse_o.mean():.4f}  median={np.median(nmse_o):.4f}")
    print(f"  NMSE modif mean={nmse_m.mean():.4f}  median={np.median(nmse_m):.4f}")
    print(f"  delta      mean={delta.mean():+.4f}  median={np.median(delta):+.4f}")

    # ── Plot 1: overlaid NMSE distributions
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    hi = float(np.percentile(np.concatenate([nmse_o, nmse_m]), 99.5))
    bins = np.linspace(0, hi, 80)
    ax.hist(nmse_o, bins=bins, alpha=0.55, label="original (3-paragraph AV)",
            color="#1f77b4", edgecolor="none")
    ax.hist(nmse_m, bins=bins, alpha=0.55, label=args.variant_label,
            color="#d62728", edgecolor="none")
    ax.axvline(nmse_o.mean(), color="#1f77b4", linestyle=":", linewidth=1.5,
                label=f"orig mean = {nmse_o.mean():.4f}")
    ax.axvline(nmse_m.mean(), color="#d62728", linestyle=":", linewidth=1.5,
                label=f"modif mean = {nmse_m.mean():.4f}")
    ax.set_xlim(0, hi)
    ax.set_xlabel("NMSE  (0 = perfect)")
    ax.set_ylabel("count")
    ax.set_title(f"NMSE distribution shift  ·  asst_content tokens  ·  n = {n:,}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    Path(args.out_dist).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dist, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {args.out_dist}")

    # ── Plot 2: per-sample Δ NMSE
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    dhi = float(np.percentile(delta, 99))
    dlo = float(np.percentile(delta, 1))
    dbins = np.linspace(dlo, dhi, 80)
    ax.hist(delta, bins=dbins, color="#444444", edgecolor="none", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.axvline(delta.mean(), color="#d62728", linestyle="--", linewidth=1.5,
                label=f"mean Δ = {delta.mean():+.4f}")
    ax.axvline(float(np.median(delta)), color="#1f77b4", linestyle="--", linewidth=1.5,
                label=f"median Δ = {np.median(delta):+.4f}")
    worse_frac = (delta > 0).mean()
    ax.set_xlabel("Δ NMSE  (modified − original)")
    ax.set_ylabel("count")
    ax.set_title(f"per-sample Δ NMSE  ·  asst_content  ·  "
                 f"{100*worse_frac:.1f}% worsen")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    Path(args.out_delta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_delta, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {args.out_delta}")


if __name__ == "__main__":
    main()
