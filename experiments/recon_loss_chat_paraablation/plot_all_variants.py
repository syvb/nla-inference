"""Combined plot: NMSE distribution overlay across all 4 variants
(original + 3 ablations), asst_content only.

Two panels written separately:
  --out-dist:  overlaid histograms of all 4 variants
  --out-bars:  mean-NMSE bar chart, per variant
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = [
    # (col, label, color)
    ("nmse_original",      "original (3-paragraph AV)",                  "#1f77b4"),
    ("nmse_final_only",    "final-¶ only (P1,P2 replaced)",              "#d62728"),
    ("nmse_removed_final", "first two only (P3 removed)",                "#2ca02c"),
    ("nmse_const_final",   "first two + canned P3 ('wide' sentence)",    "#ff7f0e"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-parquet", required=True)
    ap.add_argument("--out-dist", required=True)
    ap.add_argument("--out-bars", required=True)
    args = ap.parse_args()

    cols = ["role"] + [v[0] for v in VARIANTS]
    df = pq.read_table(args.comparison_parquet, columns=cols).to_pandas()
    df = df[df.role == "asst_content"].reset_index(drop=True)
    n = len(df)
    print(f"[plot_all] n = {n} asst_content rows")

    # ── Plot 1: overlaid distributions
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.2))
    all_vals = np.concatenate([df[c].values for c, _, _ in VARIANTS])
    hi = float(np.percentile(all_vals, 99.5))
    bins = np.linspace(0, hi, 80)
    for col, lbl, color in VARIANTS:
        vals = df[col].values
        ax.hist(vals, bins=bins, alpha=0.45, label=lbl, color=color,
                edgecolor="none")
        ax.axvline(vals.mean(), color=color, linestyle=":", linewidth=1.5)
        print(f"  {col:<22} mean={vals.mean():.4f}  median={np.median(vals):.4f}")
    ax.set_xlim(0, hi)
    ax.set_xlabel("NMSE  (0 = perfect)")
    ax.set_ylabel("count")
    ax.set_title(f"NMSE distribution — all variants  ·  asst_content  ·  n = {n:,}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    Path(args.out_dist).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dist, dpi=140)
    plt.close(fig)
    print(f"[plot_all] wrote {args.out_dist}")

    # ── Plot 2: bar chart of mean NMSE
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
    labels = [v[1] for v in VARIANTS]
    means = [df[v[0]].mean() for v in VARIANTS]
    colors = [v[2] for v in VARIANTS]
    bars = ax.bar(range(len(VARIANTS)), means, color=colors,
                   edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels],
                        fontsize=8.5)
    ax.set_ylabel("mean NMSE")
    ax.set_title(f"mean NMSE per variant  ·  asst_content  ·  n = {n:,}")
    ax.grid(alpha=0.25, linewidth=0.5, axis="y")
    for b, m in zip(bars, means):
        ax.annotate(f"{m:.4f}", xy=(b.get_x() + b.get_width() / 2, m),
                     ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_bars, dpi=140)
    plt.close(fig)
    print(f"[plot_all] wrote {args.out_bars}")


if __name__ == "__main__":
    main()
