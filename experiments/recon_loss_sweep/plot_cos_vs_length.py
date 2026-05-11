"""Plot cos vs token character length, comparing the two model runs.

Two views:
  - Per-character-length mean cos with p25-p75 IQR shading; sample counts on
    secondary axis.
  - Distribution of cos within a few representative length buckets.

Output: experiments/recon_loss_sweep/plots/cos_vs_token_length.png
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
    t = pq.read_table(parquet_path, columns=["cos", "token_str"])
    df = t.to_pandas()
    # Stripped length (drop the leading-space byte that BPE word-initial
    # tokens carry — `' the'` and `'the'` count as length 3 here).
    df["n_chars"] = df["token_str"].str.strip().str.len().fillna(0).astype(int)
    return df


def per_length(df: pd.DataFrame, max_len: int = 14) -> pd.DataFrame:
    """Aggregate cos statistics per stripped-length bucket up to max_len."""
    df = df[df["n_chars"] >= 0].copy()
    df["nb"] = df["n_chars"].clip(upper=max_len)
    g = df.groupby("nb").agg(
        n=("cos", "size"),
        mean=("cos", "mean"),
        p10=("cos", lambda x: np.percentile(x, 10)),
        p25=("cos", lambda x: np.percentile(x, 25)),
        p50=("cos", "median"),
        p75=("cos", lambda x: np.percentile(x, 75)),
        p90=("cos", lambda x: np.percentile(x, 90)),
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

    # ─── Top panel: mean cos line + p25-p75 IQR band ───────────────────────
    xs = q.index.values
    # Qwen
    ax_main.fill_between(xs, q["p25"], q["p75"], color="tab:blue", alpha=0.18,
                         label="Qwen2.5-7B p25–p75")
    ax_main.plot(xs, q["mean"], "o-", color="tab:blue", lw=2, label="Qwen2.5-7B mean")
    # Gemma
    xs2 = g.index.values
    ax_main.fill_between(xs2, g["p25"], g["p75"], color="tab:orange", alpha=0.18,
                         label="Gemma-3-12B p25–p75")
    ax_main.plot(xs2, g["mean"], "s-", color="tab:orange", lw=2,
                 label="Gemma-3-12B mean")

    ax_main.set_xlabel("Token length (chars, leading space stripped)")
    ax_main.set_ylabel("Reconstruction cosine similarity")
    ax_main.set_title("NLA round-trip cos vs. token character length\n"
                      f"20k samples each from common-pile/comma_v0.1, mean ± IQR band")
    ax_main.set_xticks(list(xs))
    xticklabels = [str(int(x)) for x in xs]
    xticklabels[-1] = f"{int(xs[-1])}+"
    ax_main.set_xticklabels(xticklabels)
    ax_main.set_ylim(0.55, 1.005)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc="lower right", framealpha=0.9)

    # Sample counts as secondary axis bars
    ax2 = ax_main.twinx()
    bar_w = 0.35
    ax2.bar(xs - bar_w / 2, q["n"], width=bar_w, alpha=0.20, color="tab:blue",
            label="Qwen n")
    ax2.bar(xs2 + bar_w / 2, g["n"], width=bar_w, alpha=0.20, color="tab:orange",
            label="Gemma n")
    ax2.set_ylabel("samples per length (bars, faint)")
    ax2.set_ylim(0, max(q["n"].max(), g["n"].max()) * 1.6)
    ax2.tick_params(axis="y", labelsize=8)

    # ─── Bottom panel: distribution shape per representative length bucket ─
    # Pick lengths that show the gradient clearly
    LEN_PICKS = [1, 2, 3, 5, 8, MAX]
    pos = np.arange(len(LEN_PICKS))
    bar_w2 = 0.36
    qwen_means  = []
    gemma_means = []
    for L in LEN_PICKS:
        qsub = qwen[qwen["n_chars"].clip(upper=MAX) == L]
        gsub = gemma[gemma["n_chars"].clip(upper=MAX) == L]
        qwen_means.append(qsub["cos"].values)
        gemma_means.append(gsub["cos"].values)

    bp_q = ax_box.boxplot(
        qwen_means, positions=pos - bar_w2 / 2, widths=bar_w2 * 0.9,
        patch_artist=True, showfliers=False,
        boxprops=dict(facecolor="tab:blue", alpha=0.5, edgecolor="tab:blue"),
        medianprops=dict(color="navy", lw=1.8),
        whiskerprops=dict(color="tab:blue"),
        capprops=dict(color="tab:blue"),
    )
    bp_g = ax_box.boxplot(
        gemma_means, positions=pos + bar_w2 / 2, widths=bar_w2 * 0.9,
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
    ax_box.set_ylabel("Cosine similarity")
    ax_box.set_title("Distribution within selected length buckets "
                     "(box = IQR, whisker = 5–95 %, fliers hidden)")
    ax_box.grid(True, axis="y", alpha=0.3)
    # Manual legend for box plot
    qbox = plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.5)
    gbox = plt.Rectangle((0, 0), 1, 1, fc="tab:orange", alpha=0.5)
    ax_box.legend([qbox, gbox], ["Qwen2.5-7B (L20)", "Gemma-3-12B (L32)"],
                  loc="lower right")
    ax_box.set_ylim(0.55, 1.005)

    out_path = OUT / "cos_vs_token_length.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")

    # Also dump the underlying numbers as a .csv for later
    csv_q = OUT / "cos_vs_length_qwen.csv"
    csv_g = OUT / "cos_vs_length_gemma.csv"
    q.to_csv(csv_q)
    g.to_csv(csv_g)
    print(f"wrote {csv_q}\nwrote {csv_g}")


if __name__ == "__main__":
    main()
