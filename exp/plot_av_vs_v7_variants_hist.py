"""Histogram of NMSE for AV vs v7 variants (PT)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def load_mse(path, col):
    t = pq.read_table(path)
    parsed = np.array(t.column("warmstart_parsed").to_pylist())
    av_p = np.array(t.column("av_parsed").to_pylist())
    sids = np.array(t.column("sample_idx").to_pylist())
    vals = np.array(t.column(col).to_pylist())
    return sids, parsed, av_p, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp/figures/av_vs_v7_variants_pt_hist.png")
    args = ap.parse_args()

    # Load all four series, restrict to rows parsed in all variants
    files = {
        "v7":      "exp/results/warmstart/warmstart_pt_v7_user_out.parquet",
        "v7p":     "exp/results/warmstart/warmstart_pt_v7plus_out.parquet",
        "v7delim": "exp/results/warmstart/warmstart_pt_v7delim_out.parquet",
    }
    tables = {k: pq.read_table(v) for k, v in files.items()}
    idx = {k: {tables[k].column("sample_idx")[i].as_py(): i for i in range(tables[k].num_rows)} for k in tables}
    shared_sids = set(idx["v7"].keys()) & set(idx["v7p"].keys()) & set(idx["v7delim"].keys())
    rows = []
    for sid in sorted(shared_sids):
        i7, ip, id_ = idx["v7"][sid], idx["v7p"][sid], idx["v7delim"][sid]
        if not (tables["v7"].column("warmstart_parsed")[i7].as_py()
                and tables["v7p"].column("warmstart_parsed")[ip].as_py()
                and tables["v7delim"].column("warmstart_parsed")[id_].as_py()
                and tables["v7"].column("av_parsed")[i7].as_py()):
            continue
        rows.append((
            tables["v7"].column("mse_av_rerun")[i7].as_py(),
            tables["v7"].column("mse_warmstart")[i7].as_py(),
            tables["v7p"].column("mse_warmstart")[ip].as_py(),
            tables["v7delim"].column("mse_warmstart")[id_].as_py(),
            tables["v7"].column("mse_empty")[i7].as_py(),
        ))
    rows = np.asarray(rows)
    av, v7, v7p, v7d, emp = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
    n = len(av)
    print(f"n_shared={n}")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    lo = max(min(av.min(), v7.min(), v7p.min(), v7d.min()), 1e-5)
    hi = max(av.max(), v7.max(), v7p.max(), v7d.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)

    series = [
        (av,  "AV (trained)",                       "#1f77b4"),
        (v7d, "v7delim (input + delim + analysis)", "#2ca02c"),
    ]
    for arr, label, color in series:
        ax.hist(arr, bins=bins, alpha=0.5, color=color,
                label=f"{label}  mean={arr.mean():.4f}  med={np.median(arr):.4f}",
                edgecolor="white", linewidth=0.4)

    ax.set_xscale("log")
    ax.set_xlabel("NMSE (mse_nrm), log scale")
    ax.set_ylabel("number of samples")
    ax.set_title(f"AV vs Sonnet 4.6 warm-start variants — NMSE distribution, PT (n={n})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
