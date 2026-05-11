"""Heatmap: NMSE by (token char-length × position quintile), Qwen vs Gemma.

NMSE = 2*(1-cos) under unit-sphere normalisation; lower is better. The two
models live in very different NMSE ranges so each panel uses its own
colour scale (annotated below the panel).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT  = ROOT / "plots"
OUT.mkdir(exist_ok=True)


def load(p):
    df = pq.read_table(p, columns=["mse_nrm", "token_str", "position"]).to_pandas()
    df["n_chars"] = df["token_str"].str.strip().str.len().fillna(0).astype(int)
    df = df.rename(columns={"mse_nrm": "nmse"})
    return df


def heatmap(ax, df, title, vmin, vmax):
    df = df[df["n_chars"] > 0].copy()
    df["len_b"] = pd.cut(df["n_chars"], [0, 1, 2, 3, 5, 8, 999],
                         labels=["1", "2", "3", "4-5", "6-8", "9+"])
    df["pos_b"] = pd.qcut(
        df["position"], 5,
        labels=["pos 0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    )
    M = df.groupby(["len_b", "pos_b"], observed=True)["nmse"].mean().unstack()
    # Use reversed RdYlGn so low NMSE = green, high NMSE = red.
    im = ax.imshow(M.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(M.columns)))
    ax.set_xticklabels(M.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(M.index)))
    ax.set_yticklabels(M.index)
    ax.set_xlabel("Position quintile in sequence")
    ax.set_ylabel("Token char length (stripped)")
    ax.set_title(title)
    midpoint = (vmin + vmax) / 2
    for i in range(len(M.index)):
        for j in range(len(M.columns)):
            v = M.values[i, j]
            color = "white" if v > midpoint else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=color, fontsize=9)
    return im, M


def main():
    qwen  = load(DATA / "results_qwen7_20000.parquet")
    gemma = load(DATA / "results_20000.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6),
                              gridspec_kw={"wspace": 0.32})
    # Per-panel colour ranges so each model's gradient is visible.
    im_q, M_q = heatmap(axes[0], qwen,
                        "Qwen2.5-7B (L20)\nmean NMSE = 0.249",
                        vmin=0.18, vmax=0.30)
    im_g, M_g = heatmap(axes[1], gemma,
                        "Gemma-3-12B (L32)\nmean NMSE = 0.014",
                        vmin=0.005, vmax=0.025)

    fig.colorbar(im_q, ax=axes[0], fraction=0.042, pad=0.02,
                 label="NMSE (Qwen)")
    fig.colorbar(im_g, ax=axes[1], fraction=0.042, pad=0.02,
                 label="NMSE (Gemma)")

    fig.suptitle("NMSE = 2·(1 − cos) by token length × position-in-sequence "
                 "(per-panel colour scale)",
                  fontsize=12.5, y=1.02)

    out = OUT / "nmse_heatmap_length_position.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
