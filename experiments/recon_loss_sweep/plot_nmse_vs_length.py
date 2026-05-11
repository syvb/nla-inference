"""Plot NMSE vs token character length, comparing the two model runs.

NMSE = mse_nrm column from the results parquet, == 2*(1-cos) under the
unit-sphere normalisation used by NLACritic. Range [0, 4]; 0 = perfect,
2 = orthogonal. Lower is better.

Two views:
  - Top: per-character-length mean NMSE with p25-p75 IQR shading on log-y;
    sample counts as faint background bars.
  - Bottom: distribution box plots within representative length buckets.

Output: experiments/recon_loss_sweep/plots/nmse_vs_token_length.png
"""
from __future__ import annotations

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


def load(parquet_path: Path) -> pd.DataFrame:
    t = pq.read_table(parquet_path, columns=["mse_nrm", "token_str"])
    df = t.to_pandas()
    df["n_chars"] = df["token_str"].str.strip().str.len().fillna(0).astype(int)
    df = df.rename(columns={"mse_nrm": "nmse"})
    return df


def per_length(df: pd.DataFrame, max_len: int = 14) -> pd.DataFrame:
    df = df[df["n_chars"] >= 0].copy()
    df["nb"] = df["n_chars"].clip(upper=max_len)
    g = df.groupby("nb").agg(
        n=("nmse", "size"),
        mean=("nmse", "mean"),
        p10=("nmse", lambda x: np.percentile(x, 10)),
        p25=("nmse", lambda x: np.percentile(x, 25)),
        p50=("nmse", "median"),
        p75=("nmse", lambda x: np.percentile(x, 75)),
        p90=("nmse", lambda x: np.percentile(x, 90)),
    )
    return g


def main():
    qwen  = load(DATA / "results_qwen7_20000.parquet")
    gemma = load(DATA / "results_20000.parquet")
    print(f"loaded qwen={len(qwen)}, gemma={len(gemma)}")

    MAX = 14
    q = per_length(qwen, MAX)
    g = per_length(gemma, MAX)

    fig, (ax_main, ax_box) = plt.subplots(
        2, 1, figsize=(11, 9),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.35},
    )

    # ─── Top panel: NMSE on log-y, mean line + IQR band ────────────────────
    xs = q.index.values
    ax_main.fill_between(xs, q["p25"], q["p75"], color="tab:blue", alpha=0.18,
                         label="Qwen2.5-7B p25–p75")
    ax_main.plot(xs, q["mean"], "o-", color="tab:blue", lw=2,
                 label="Qwen2.5-7B mean")
    xs2 = g.index.values
    ax_main.fill_between(xs2, g["p25"], g["p75"], color="tab:orange", alpha=0.18,
                         label="Gemma-3-12B p25–p75")
    ax_main.plot(xs2, g["mean"], "s-", color="tab:orange", lw=2,
                 label="Gemma-3-12B mean")

    ax_main.set_yscale("log")
    ax_main.set_xlabel("Token length (chars, leading space stripped)")
    ax_main.set_ylabel("NMSE = 2·(1 − cos)   (log scale, lower = better)")
    ax_main.set_title("NLA round-trip NMSE vs. token character length\n"
                      "20k samples each from common-pile/comma_v0.1, mean ± IQR band")
    ax_main.set_xticks(list(xs))
    xticklabels = [str(int(x)) for x in xs]
    xticklabels[-1] = f"{int(xs[-1])}+"
    ax_main.set_xticklabels(xticklabels)
    # Reasonable log-y range covering both
    ax_main.set_ylim(0.002, 1.0)
    ax_main.grid(True, which="both", alpha=0.3)
    ax_main.legend(loc="upper right", framealpha=0.9)

    # Sample counts as secondary axis bars (linear)
    ax2 = ax_main.twinx()
    bar_w = 0.35
    ax2.bar(xs - bar_w / 2, q["n"], width=bar_w, alpha=0.18, color="tab:blue")
    ax2.bar(xs2 + bar_w / 2, g["n"], width=bar_w, alpha=0.18, color="tab:orange")
    ax2.set_ylabel("samples per length (faint bars)")
    ax2.set_ylim(0, max(q["n"].max(), g["n"].max()) * 1.6)
    ax2.tick_params(axis="y", labelsize=8)

    # Reference annotations
    ax_main.axhline(0.02, color="gray", lw=0.6, ls=":", alpha=0.7)
    ax_main.text(13.5, 0.022, "≈ cos 0.99", fontsize=8, color="gray", ha="right")
    ax_main.axhline(0.20, color="gray", lw=0.6, ls=":", alpha=0.7)
    ax_main.text(13.5, 0.215, "≈ cos 0.90", fontsize=8, color="gray", ha="right")
    ax_main.axhline(2.0, color="red",  lw=0.6, ls="--", alpha=0.4)

    # ─── Bottom panel: distribution shape per representative length bucket ─
    LEN_PICKS = [1, 2, 3, 5, 8, MAX]
    pos = np.arange(len(LEN_PICKS))
    bar_w2 = 0.36
    qwen_vals  = []
    gemma_vals = []
    for L in LEN_PICKS:
        qsub = qwen[qwen["n_chars"].clip(upper=MAX) == L]["nmse"].values
        gsub = gemma[gemma["n_chars"].clip(upper=MAX) == L]["nmse"].values
        qwen_vals.append(qsub)
        gemma_vals.append(gsub)

    ax_box.boxplot(
        qwen_vals, positions=pos - bar_w2 / 2, widths=bar_w2 * 0.9,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor="tab:blue", alpha=0.5, edgecolor="tab:blue"),
        medianprops=dict(color="navy", lw=1.8),
        whiskerprops=dict(color="tab:blue"),
        capprops=dict(color="tab:blue"),
    )
    ax_box.boxplot(
        gemma_vals, positions=pos + bar_w2 / 2, widths=bar_w2 * 0.9,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor="tab:orange", alpha=0.5, edgecolor="tab:orange"),
        medianprops=dict(color="darkred", lw=1.8),
        whiskerprops=dict(color="tab:orange"),
        capprops=dict(color="tab:orange"),
    )
    labels = [str(L) if L < MAX else f"{L}+" for L in LEN_PICKS]
    ax_box.set_xticks(pos)
    ax_box.set_xticklabels(labels)
    ax_box.set_xlabel("Token length (chars)")
    ax_box.set_ylabel("NMSE (log scale)")
    ax_box.set_yscale("log")
    ax_box.set_ylim(0.002, 1.0)
    ax_box.set_title("Distribution within selected length buckets "
                     "(box = IQR, whisker = 5–95 %, fliers hidden)")
    ax_box.grid(True, which="both", axis="y", alpha=0.3)
    qbox = plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.5)
    gbox = plt.Rectangle((0, 0), 1, 1, fc="tab:orange", alpha=0.5)
    ax_box.legend([qbox, gbox], ["Qwen2.5-7B (L20)", "Gemma-3-12B (L32)"],
                  loc="upper right")

    out_path = OUT / "nmse_vs_token_length.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")

    # Dump tables
    q.to_csv(OUT / "nmse_vs_length_qwen.csv")
    g.to_csv(OUT / "nmse_vs_length_gemma.csv")


if __name__ == "__main__":
    main()
