"""Histogram of NMSE for AV (baseline) vs av_delim (AR with AV expl + source text)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-out", default="exp/results/warmstart/warmstart_pt_v7_user_out.parquet",
                    help="parquet with mse_av_rerun + mse_empty (anchor for sample_idx and empty baseline)")
    ap.add_argument("--avdelim-out", default="exp/results/warmstart/warmstart_pt_av_delim_out.parquet")
    ap.add_argument("--out", default="exp/figures/av_vs_avdelim_pt_hist.png")
    args = ap.parse_args()

    a = pq.read_table(args.av_out)
    d = pq.read_table(args.avdelim_out)

    a_idx = {a.column("sample_idx")[i].as_py(): i for i in range(a.num_rows)}
    d_idx = {d.column("sample_idx")[i].as_py(): i for i in range(d.num_rows)}
    shared = sorted(set(a_idx) & set(d_idx))

    rows = []
    for sid in shared:
        ia, id_ = a_idx[sid], d_idx[sid]
        if not (a.column("av_parsed")[ia].as_py()
                and d.column("warmstart_parsed")[id_].as_py()):
            continue
        rows.append((
            a.column("mse_av_rerun")[ia].as_py(),
            d.column("mse_warmstart")[id_].as_py(),
        ))
    rows = np.asarray(rows)
    av, avd = rows[:, 0], rows[:, 1]
    n = len(av)
    print(f"n_shared={n}")
    print(f"  AV       mean={av.mean():.4f}  med={np.median(av):.4f}")
    print(f"  av_delim mean={avd.mean():.4f}  med={np.median(avd):.4f}")
    print(f"  av_delim beats AV on {(avd < av).sum()}/{n} = {100*(avd<av).sum()/n:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    lo = max(min(av.min(), avd.min()), 1e-5)
    hi = max(av.max(), avd.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)

    ax.hist(av,  bins=bins, alpha=0.55, color="#1f77b4",
            label=f"AV baseline                mean={av.mean():.4f}  med={np.median(av):.4f}",
            edgecolor="white", linewidth=0.5)
    ax.hist(avd, bins=bins, alpha=0.55, color="#2ca02c",
            label=f"av_delim (AR + AV expl + source)  mean={avd.mean():.4f}  med={np.median(avd):.4f}",
            edgecolor="white", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("NMSE (mse_nrm), log scale")
    ax.set_ylabel("number of samples")
    ax.set_title(f"AV baseline vs AR-with-source-text-and-AV-explanation, PT (n={n})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
