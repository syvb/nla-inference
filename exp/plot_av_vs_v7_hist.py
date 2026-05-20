"""Histogram of NMSE for AV vs v7 (Sonnet 4.6 with user system prompt) on PT."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="exp/results/warmstart/warmstart_pt_v7_user_out.parquet")
    ap.add_argument("--out", default="exp/figures/av_vs_v7_pt_hist.png")
    args = ap.parse_args()

    t = pq.read_table(args.input)
    parsed = np.array(t.column("warmstart_parsed").to_pylist())
    av_p = np.array(t.column("av_parsed").to_pylist())
    av = np.array(t.column("mse_av_rerun").to_pylist())
    ws = np.array(t.column("mse_warmstart").to_pylist())
    mask = parsed & av_p & ~np.isnan(av) & ~np.isnan(ws)
    av, ws = av[mask], ws[mask]
    n = len(av)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Shared log-spaced bins
    lo = max(min(av.min(), ws.min()), 1e-5)
    hi = max(av.max(), ws.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)

    ax.hist(av, bins=bins, alpha=0.55, color="#1f77b4",
            label=f"AV (trained)  mean={av.mean():.4f}  median={np.median(av):.4f}",
            edgecolor="white", linewidth=0.5)
    ax.hist(ws, bins=bins, alpha=0.55, color="#d62728",
            label=f"v7: Sonnet 4.6 + user prompt  mean={ws.mean():.4f}  median={np.median(ws):.4f}",
            edgecolor="white", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("NMSE (mse_nrm), log scale")
    ax.set_ylabel("number of samples")
    ax.set_title(f"AV vs Sonnet 4.6 (v7 user prompt) — NMSE distribution, PT (n={n})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
