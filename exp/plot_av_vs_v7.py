"""Plot error distributions for AV vs v7 (Sonnet 4.6 with user system prompt) on PT.

Two-panel figure:
  left  — ECDF of mse_nrm for AV, v7 warm-start, and empty baseline
  right — paired per-row scatter, AV mse vs v7 mse, with y=x diagonal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="exp/results/warmstart/warmstart_pt_v7_user_out.parquet")
    ap.add_argument("--out", default="exp/results/warmstart/av_vs_v7_pt.png")
    args = ap.parse_args()

    t = pq.read_table(args.input)
    parsed = np.array(t.column("warmstart_parsed").to_pylist())
    av_p = np.array(t.column("av_parsed").to_pylist())
    av = np.array(t.column("mse_av_rerun").to_pylist())
    ws = np.array(t.column("mse_warmstart").to_pylist())
    emp = np.array(t.column("mse_empty").to_pylist())
    mask = parsed & av_p & ~np.isnan(av) & ~np.isnan(ws) & ~np.isnan(emp)
    av, ws, emp = av[mask], ws[mask], emp[mask]
    n = len(av)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ─── Left: ECDF ───────────────────────────────────────────────────
    def ecdf(x):
        s = np.sort(x); return s, np.arange(1, len(s) + 1) / len(s)

    for arr, label, color in [
        (av,  f"AV (trained)",                "#1f77b4"),
        (ws,  f"v7: Sonnet 4.6 + user prompt", "#d62728"),
        (emp, f"empty prompt",                  "#7f7f7f"),
    ]:
        x, y = ecdf(arr)
        ax1.plot(x, y, label=f"{label}  (mean={arr.mean():.4f})", linewidth=2, color=color)

    ax1.set_xscale("log")
    ax1.set_xlabel("mse_nrm (log scale)")
    ax1.set_ylabel("cumulative fraction of samples")
    ax1.set_title(f"ECDF of reconstruction error (PT, n={n})")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    # ─── Right: paired scatter ────────────────────────────────────────
    ax2.scatter(av, ws, s=10, alpha=0.4, color="#2ca02c", edgecolors="none")
    lim_lo = min(av.min(), ws.min()) * 0.8
    lim_hi = max(av.max(), ws.max()) * 1.2
    ax2.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1, alpha=0.5, label="y=x")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(lim_lo, lim_hi); ax2.set_ylim(lim_lo, lim_hi)
    ax2.set_xlabel("AV mse_nrm")
    ax2.set_ylabel("v7 mse_nrm")
    n_av_better = (av < ws).sum()
    ax2.set_title(f"Per-row paired: AV better on {n_av_better}/{n} ({100*n_av_better/n:.0f}%)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    fig.suptitle("AV vs Sonnet 4.6 (v7 system prompt) — reconstruction error, PT dataset",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e3:.0f} KB)")

    # Print summary alongside
    def fve(m, e): return ((e - m) / e).mean()
    print()
    print(f"n_paired = {n}")
    print(f"  AV   mean={av.mean():.4f}  med={np.median(av):.4f}  FVE={fve(av, emp):+.4f}")
    print(f"  v7   mean={ws.mean():.4f}  med={np.median(ws):.4f}  FVE={fve(ws, emp):+.4f}")
    print(f"  emp  mean={emp.mean():.4f}  med={np.median(emp):.4f}")
    print(f"  AV better on {n_av_better}/{n} rows ({100*n_av_better/n:.1f}%)")


if __name__ == "__main__":
    main()
