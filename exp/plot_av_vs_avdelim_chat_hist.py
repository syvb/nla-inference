"""Histogram of NMSE for AV baseline vs av_delim on the full chat dataset.

Reads:
  - warmstart_chat_av_delim_20k_out.parquet (av_delim mse, empty mse)
  - results_chat_20k.parquet (AV saved recon + activations) to compute AV baseline mse

Computes AV baseline mse on the fly from the saved recon vectors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def mse_nrm_batch(preds, golds, s):
    pn = preds / (np.linalg.norm(preds, axis=-1, keepdims=True) + 1e-12) * s
    gn = golds / (np.linalg.norm(golds, axis=-1, keepdims=True) + 1e-12) * s
    return ((pn - gn) ** 2).mean(axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-delim", default="exp/results/warmstart/warmstart_chat_av_delim_20k_out.parquet")
    ap.add_argument("--results", default="exp/results/warmstart/results_chat_20k.parquet")
    ap.add_argument("--out", default="exp/figures/av_vs_avdelim_chat_hist.png")
    ap.add_argument("--mse-scale", type=float, default=61.96773353931867)
    args = ap.parse_args()

    print("[load] av_delim parquet")
    d = pq.read_table(args.av_delim)
    d_idx = {d.column("sample_idx")[i].as_py(): i for i in range(d.num_rows)}

    print("[load] results parquet (for activation + recon)")
    r = pq.read_table(args.results, columns=["sample_idx", "activation", "recon", "av_parsed"])
    # Compute AV baseline mse from saved recon
    n_r = r.num_rows
    acts = np.asarray(r.column("activation").combine_chunks().values).astype(np.float32).reshape(n_r, -1)
    recs = np.asarray(r.column("recon").combine_chunks().values).astype(np.float32).reshape(n_r, -1)
    av_parsed = np.asarray(r.column("av_parsed").to_pylist())
    av_mse = mse_nrm_batch(recs, acts, args.mse_scale)
    print(f"  av baseline mean mse (all parsed): {av_mse[av_parsed].mean():.4f}  n={av_parsed.sum()}")

    r_idx = {r.column("sample_idx")[i].as_py(): i for i in range(n_r)}

    # Build paired arrays for rows present + parsed in both
    rows = []
    for i in range(d.num_rows):
        sid = d.column("sample_idx")[i].as_py()
        if not d.column("warmstart_parsed")[i].as_py(): continue
        if sid not in r_idx: continue
        ir = r_idx[sid]
        if not av_parsed[ir]: continue
        rows.append((av_mse[ir], d.column("mse_warmstart")[i].as_py(), d.column("mse_empty")[i].as_py()))
    rows = np.asarray(rows)
    av, avd, emp = rows[:, 0], rows[:, 1], rows[:, 2]
    n = len(av)
    print(f"n_paired={n}")
    print(f"  AV baseline mean={av.mean():.4f}  med={np.median(av):.4f}")
    print(f"  av_delim    mean={avd.mean():.4f}  med={np.median(avd):.4f}")
    print(f"  av_delim beats AV on {(avd < av).sum()}/{n} = {100*(avd<av).sum()/n:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    lo = max(min(av.min(), avd.min()), 1e-5)
    hi = max(av.max(), avd.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)

    ax.hist(av,  bins=bins, alpha=0.55, color="#1f77b4",
            label=f"AV baseline                mean={av.mean():.4f}  med={np.median(av):.4f}",
            edgecolor="white", linewidth=0.3)
    ax.hist(avd, bins=bins, alpha=0.55, color="#2ca02c",
            label=f"av_delim (AR + AV expl + source)  mean={avd.mean():.4f}  med={np.median(avd):.4f}",
            edgecolor="white", linewidth=0.3)

    ax.set_xscale("log")
    ax.set_xlabel("NMSE (mse_nrm), log scale")
    ax.set_ylabel("number of samples")
    ax.set_title(f"AV baseline vs AR-with-source-text-and-AV-explanation, CHAT (n={n})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
