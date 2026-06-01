"""Figure for the residual-refresh AV probe (FINDINGS §5c).

Reads the scored variant parquets in exp/results/warmstart/refresh_scored/ and
plots, against the refresh strength alpha (peak fraction of inj_scale added to
each output token's residual):
  (A) parse/coverage rate -- the fraction of the 500-sample probe that still
      produces a well-formed <explanation>...</explanation>;
  (B) paired Delta-MSE (variant - baseline) on the samples parsed in BOTH the
      variant and the alpha=0 baseline -- the bias-free measure of whether the
      refresh helps reconstruction.

Story: any alpha weak enough to keep coverage ~100% leaves Delta-MSE ~ 0 (no
information added); any alpha strong enough to register collapses coverage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

SCORED = Path("exp/results/warmstart/refresh_scored")
BASE = "refresh_a0_n500"           # alpha=0, warm4/tau8 manual-loop baseline
# alpha -> scored-file stem, all warm=4 tau=8 (the canonical decay schedule)
SERIES = [
    (0.0005, "refresh_a0.0005"),
    (0.002,  "refresh_a0.002"),
    (0.005,  "ref_a0.005_w4t8"),
    (0.01,   "refresh_a0.01"),
    (0.05,   "refresh_a0.05"),
    (0.2,    "refresh_a0.2"),
    (1.0,    "refresh_a1.0"),
]
INJ_SCALE, TOK_NORM = 80000.0, 60.0  # added-norm = alpha*INJ_SCALE; token embed ~60


def load(stem):
    t = pq.read_table(SCORED / f"{stem}_scored.parquet")
    sid = t.column("sample_idx").to_pylist()
    mse = np.asarray(t.column("mse_warmstart"))
    parsed = np.asarray(t.column("warmstart_parsed"))
    n = len(sid)
    valid = {int(s): float(m) for s, m, p in zip(sid, mse, parsed) if p and m == m}
    return valid, n


base, N = load(BASE)
alphas, cover, dmean, paired_n = [], [], [], []
for a, stem in SERIES:
    var, _ = load(stem)
    common = sorted(set(base) & set(var))
    b = np.array([base[s] for s in common]); v = np.array([var[s] for s in common])
    alphas.append(a)
    cover.append(100.0 * len(var) / N)
    dmean.append(float(v.mean() - b.mean()))
    paired_n.append(len(common))

alphas = np.array(alphas); cover = np.array(cover)
dmean = np.array(dmean); paired_n = np.array(paired_n)
base_cover = 100.0 * len(base) / N

fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)

# Panel A: coverage
axA.axhline(base_cover, color="green", ls="--", lw=1.2, label=f"alpha=0 baseline ({base_cover:.0f}%)")
axA.plot(alphas, cover, "o-", color="C3", lw=2, ms=7)
axA.set_xscale("log")
axA.set_ylabel("coverage:\nwell-formed output (%)")
axA.set_ylim(0, 105)
axA.set_title("Residual-refresh of the activation into output tokens (gemma3-12b AV, n=500)")
axA.grid(alpha=0.3); axA.legend(loc="lower left", fontsize=9)
axA.annotate("safe but inert", xy=(0.0012, 102), fontsize=9, color="gray", ha="center")
axA.annotate("coverage collapse", xy=(0.05, 55), fontsize=9, color="C3", ha="center")

# Panel B: paired Delta-MSE on common samples
colors = ["C0" if c > 95 else "0.7" for c in cover]
axB.axhline(0, color="green", ls="--", lw=1.2, label="no change vs baseline")
for a, d, c, pn in zip(alphas, dmean, colors, paired_n):
    axB.plot(a, d, "o", color=c, ms=7)
axB.plot(alphas, dmean, "-", color="0.6", lw=1, alpha=0.5)
axB.set_xscale("log")
axB.set_xlabel("alpha  (peak refresh strength = fraction of inj_scale; added-norm = alpha x 80000)")
axB.set_ylabel("paired delta-MSE\n(variant - baseline,\nlower=better)")
axB.grid(alpha=0.3); axB.legend(loc="upper left", fontsize=9)
axB.annotate("full-coverage points: delta ~ 0\n(win-rate ~0.51)", xy=(0.0009, dmean[0]),
             xytext=(0.0009, max(dmean) * 0.6 + 0.0003), fontsize=8.5, color="C0", ha="left")
axB.annotate("faded = low coverage\n(selection-biased)", xy=(0.1, 0), xytext=(0.1, min(dmean) * 0.6 - 0.0002),
             fontsize=8.5, color="0.5", ha="center")

# secondary top axis: added-norm relative to token-embedding norm (~60)
def a2ratio(a):
    return a * INJ_SCALE / TOK_NORM
secax = axA.secondary_xaxis("top", functions=(a2ratio, lambda r: r * TOK_NORM / INJ_SCALE))
secax.set_xlabel("added-vector norm  /  output-token-embedding norm  (x)")

fig.tight_layout()
out = Path("exp/figures/av_refresh_fve.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
for a, c, d, pn in zip(alphas, cover, dmean, paired_n):
    print(f"  alpha={a:<7} coverage={c:5.1f}%  paired_n={pn:3d}  dMSE={d:+.4f}")
