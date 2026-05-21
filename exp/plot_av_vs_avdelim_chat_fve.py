"""Histogram of per-sample FVE for AV baseline vs av_delim on chat 20k.

FVE = (mse_empty - mse_variant) / mse_empty per sample. Higher is better.
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
    ap.add_argument("--out", default="exp/figures/av_vs_avdelim_chat_fve.png")
    ap.add_argument("--mse-scale", type=float, default=61.96773353931867)
    args = ap.parse_args()

    print("[load] av_delim parquet")
    d = pq.read_table(args.av_delim)
    print("[load] results parquet (for activation + recon)")
    r = pq.read_table(args.results, columns=["sample_idx", "activation", "recon", "av_parsed"])

    n_r = r.num_rows
    acts = np.asarray(r.column("activation").combine_chunks().values).astype(np.float32).reshape(n_r, -1)
    recs = np.asarray(r.column("recon").combine_chunks().values).astype(np.float32).reshape(n_r, -1)
    av_parsed = np.asarray(r.column("av_parsed").to_pylist())
    av_mse_all = mse_nrm_batch(recs, acts, args.mse_scale)

    # Baseline: predict the mean activation across the dataset (constant prediction).
    # Compute mean from all parsed-AV activations, then per-sample mse_nrm(mean, sample).
    mean_act = acts[av_parsed].mean(axis=0)
    mean_pred = np.broadcast_to(mean_act, acts.shape)
    mean_mse_all = mse_nrm_batch(mean_pred, acts, args.mse_scale)
    print(f"  mean-activation baseline mse: mean={mean_mse_all[av_parsed].mean():.4f} med={np.median(mean_mse_all[av_parsed]):.4f}")

    r_idx = {r.column("sample_idx")[i].as_py(): i for i in range(n_r)}
    rows = []
    for i in range(d.num_rows):
        sid = d.column("sample_idx")[i].as_py()
        if not d.column("warmstart_parsed")[i].as_py(): continue
        if sid not in r_idx: continue
        ir = r_idx[sid]
        if not av_parsed[ir]: continue
        rows.append((av_mse_all[ir], d.column("mse_warmstart")[i].as_py(), mean_mse_all[ir]))
    rows = np.asarray(rows)
    av, avd, base = rows[:, 0], rows[:, 1], rows[:, 2]

    # Per-sample FVE (for visualization).
    fve_av  = 1 - av  / base
    fve_avd = 1 - avd / base
    # Paper formula: aggregate FVE = 1 - mean(loss) / mean(var)  (single scalar)
    fve_agg_av  = 1 - av.mean()  / base.mean()
    fve_agg_avd = 1 - avd.mean() / base.mean()
    n = len(av)
    print(f"n_paired={n}")
    print(f"  AV       per-sample FVE mean={fve_av.mean():.4f}  med={np.median(fve_av):.4f}  aggregate={fve_agg_av:.4f}")
    print(f"  av_delim per-sample FVE mean={fve_avd.mean():.4f}  med={np.median(fve_avd):.4f}  aggregate={fve_agg_avd:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # FVE is unbounded above (can be negative if variant > empty); clip x for display.
    lo, hi = -0.2, 1.0
    bins = np.linspace(lo, hi, 80)
    ax.hist(np.clip(fve_av,  lo, hi), bins=bins, alpha=0.55, color="#1f77b4",
            label=f"AV baseline   aggregate FVE = {fve_agg_av:.4f}  (paper formula)",
            edgecolor="white", linewidth=0.3)
    ax.hist(np.clip(fve_avd, lo, hi), bins=bins, alpha=0.55, color="#2ca02c",
            label=f"av_delim       aggregate FVE = {fve_agg_avd:.4f}  (paper formula)",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("per-sample FVE  = 1 − ‖h_i − ĥ_i‖² / ‖h_i − h̄‖²")
    ax.set_ylabel("number of samples")
    ax.set_title(f"AV baseline vs av_delim — FVE distribution, CHAT (n={n})\n"
                 f"FVE=0 ↔ predicting the mean activation; FVE=1 ↔ perfect reconstruction")
    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(lo, hi)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
