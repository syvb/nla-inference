"""Analyze a results parquet from recon_loss_sweep.py decode.

Reports:
  - Overall MSE / cos distribution (mean, median, percentiles)
  - AV parse rate
  - Top-50 high-cos tokens (best reconstructed) and bottom-50 low-cos tokens
    (worst reconstructed), filtered to tokens seen ≥ MIN_COUNT times
  - Cos as a function of:
      * vec_norm bucket (does an unusually high-norm activation reconstruct
        worse, as the README warns?)
      * position bucket (early positions are noisy per README — does this
        survive after the skip_first=10 cutoff?)
      * seq_len bucket
  - A handful of qualitative examples per bucket

Usage:
    python analyze.py results.parquet [--min-count 5] [--out report.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def pct(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p))


def fmt_token(s: str) -> str:
    """Render a token string with whitespace/control chars made visible."""
    if not isinstance(s, str):
        return repr(s)
    return repr(s)[1:-1]  # drop surrounding quotes from repr


def bucket_by(values: np.ndarray, n_buckets: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Equal-frequency buckets. Returns (bucket_idx[N], edges[n_buckets+1])."""
    edges = np.quantile(values, np.linspace(0, 1, n_buckets + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_buckets - 1)
    return idx, edges


def section(title: str) -> str:
    return f"\n## {title}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--min-count", type=int, default=5,
                    help="Min occurrences for a token to appear in token tables.")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", default="-",
                    help="Write report to this path; '-' = stdout.")
    args = ap.parse_args()

    print(f"[load] {args.parquet}")
    t = pq.read_table(args.parquet, columns=[
        "sample_idx", "token_id", "token_str", "position", "seq_len",
        "vec_norm", "text_preview", "explanation", "av_parsed",
        "mse_nrm", "cos",
    ])
    df = t.to_pandas()
    n = len(df)
    print(f"[load] {n} rows")

    out_lines: list[str] = []
    w = out_lines.append

    w(f"# Reconstruction-loss sweep — `{Path(args.parquet).name}`\n")
    w(f"**N = {n}** activations.")
    w(f"AV parse rate: **{df['av_parsed'].mean():.3f}** "
      f"({int(df['av_parsed'].sum())} / {n})\n")

    cos = df["cos"].to_numpy()
    mse = df["mse_nrm"].to_numpy()

    w(section("Overall MSE / cos"))
    w("| metric | mean | p10 | p50 | p90 | p99 |")
    w("|---|---|---|---|---|---|")
    for name, x in [("cos", cos), ("mse_nrm", mse)]:
        w(f"| {name} | {x.mean():.3f} | {pct(x,10):.3f} | {pct(x,50):.3f} "
          f"| {pct(x,90):.3f} | {pct(x,99):.3f} |")

    # ─── Per-token aggregation ───
    grp = df.groupby("token_str", observed=True).agg(
        n=("cos", "size"),
        cos_mean=("cos", "mean"),
        mse_mean=("mse_nrm", "mean"),
        norm_mean=("vec_norm", "mean"),
    ).reset_index()
    grp = grp[grp["n"] >= args.min_count]
    grp = grp.sort_values("cos_mean", ascending=False)

    w(section(f"Top {args.top} BEST-reconstructed tokens "
              f"(highest cos, n ≥ {args.min_count})"))
    w("| rank | token | n | cos_mean | mse_mean | norm_mean |")
    w("|---|---|---|---|---|---|")
    for i, row in enumerate(grp.head(args.top).itertuples(), 1):
        w(f"| {i} | `{fmt_token(row.token_str)}` | {row.n} "
          f"| {row.cos_mean:.3f} | {row.mse_mean:.3f} | {row.norm_mean:.1f} |")

    w(section(f"Bottom {args.top} WORST-reconstructed tokens "
              f"(lowest cos, n ≥ {args.min_count})"))
    w("| rank | token | n | cos_mean | mse_mean | norm_mean |")
    w("|---|---|---|---|---|---|")
    for i, row in enumerate(grp.tail(args.top)[::-1].itertuples(), 1):
        w(f"| {i} | `{fmt_token(row.token_str)}` | {row.n} "
          f"| {row.cos_mean:.3f} | {row.mse_mean:.3f} | {row.norm_mean:.1f} |")

    # ─── Cos vs vec_norm ───
    w(section("Cos by activation L2-norm bucket (decile)"))
    norms = df["vec_norm"].to_numpy()
    bidx, edges = bucket_by(norms, 10)
    w("| bucket | norm_range | n | cos_mean | cos_median |")
    w("|---|---|---|---|---|")
    for b in range(10):
        m = bidx == b
        w(f"| {b} | {edges[b]:.0f} – {edges[b+1]:.0f} | {m.sum()} "
          f"| {cos[m].mean():.3f} | {np.median(cos[m]):.3f} |")

    # ─── Cos vs position ───
    w(section("Cos by token position bucket (decile, all positions ≥ 10)"))
    pos = df["position"].to_numpy()
    bidx, edges = bucket_by(pos.astype(float), 10)
    w("| bucket | pos_range | n | cos_mean |")
    w("|---|---|---|---|")
    for b in range(10):
        m = bidx == b
        w(f"| {b} | {int(edges[b])} – {int(edges[b+1])} | {m.sum()} "
          f"| {cos[m].mean():.3f} |")

    # ─── Cos vs seq_len ───
    w(section("Cos by sequence length bucket (decile)"))
    sl = df["seq_len"].to_numpy()
    bidx, edges = bucket_by(sl.astype(float), 10)
    w("| bucket | seq_len_range | n | cos_mean |")
    w("|---|---|---|---|")
    for b in range(10):
        m = bidx == b
        w(f"| {b} | {int(edges[b])} – {int(edges[b+1])} | {m.sum()} "
          f"| {cos[m].mean():.3f} |")

    # ─── Qualitative examples ───
    w(section("Qualitative — 10 best-reconstructed individual samples"))
    for r in df.nlargest(10, "cos").itertuples():
        w(f"- cos={r.cos:.3f} pos={r.position} ‖v‖={r.vec_norm:.1f} "
          f"tok=`{fmt_token(r.token_str)}`  expl=`{(r.explanation or '')[:160]}`")

    w(section("Qualitative — 10 worst-reconstructed individual samples"))
    for r in df.nsmallest(10, "cos").itertuples():
        w(f"- cos={r.cos:.3f} pos={r.position} ‖v‖={r.vec_norm:.1f} "
          f"tok=`{fmt_token(r.token_str)}`  expl=`{(r.explanation or '')[:160]}`")

    w(section("Qualitative — 10 random samples"))
    for r in df.sample(10, random_state=0).itertuples():
        w(f"- cos={r.cos:.3f} pos={r.position} ‖v‖={r.vec_norm:.1f} "
          f"tok=`{fmt_token(r.token_str)}`  expl=`{(r.explanation or '')[:160]}`")

    text = "\n".join(out_lines)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
