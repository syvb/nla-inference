"""Histogram: AV on activation (baseline) vs AV on text (this experiment), chat 5k."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="exp/results/warmstart/warmstart_av_on_text_5k_out.parquet")
    ap.add_argument("--out", default="exp/figures/av_on_text_vs_activation_hist.png")
    args = ap.parse_args()

    t = pq.read_table(args.inp)
    parsed = np.array(t.column("warmstart_parsed").to_pylist())
    avt = np.array(t.column("mse_warmstart").to_pylist())
    av_act = np.array(t.column("mse_av_saved").to_pylist())
    m = parsed & ~np.isnan(avt) & ~np.isnan(av_act)
    avt, av_act = avt[m], av_act[m]
    n = len(avt)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    lo = max(min(avt.min(), av_act.min()), 1e-5)
    hi = max(avt.max(), av_act.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 70)
    ax.hist(av_act, bins=bins, alpha=0.55, color="#1f77b4",
            label=f"AV on activation (baseline)  mean={av_act.mean():.4f}  med={np.median(av_act):.4f}",
            edgecolor="white", linewidth=0.3)
    ax.hist(avt, bins=bins, alpha=0.55, color="#d62728",
            label=f"AV on TEXT (this experiment)  mean={avt.mean():.4f}  med={np.median(avt):.4f}",
            edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("NMSE (mse_nrm), log scale")
    ax.set_ylabel("number of samples")
    ax.set_title(f"AV given the activation vs AV given the source text, CHAT (n={n})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
