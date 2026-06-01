"""Figure for the attention-bias x K-repeat probe (FINDINGS §5c round 6).

Reads the scored grid in exp/results/warmstart/attnbias_scored/ and plots, vs the
attention-logit bias b applied to the K marker key positions:
  (A) paired win-rate -- fraction of matched samples (parsed in BOTH the cell and
      the b=0,K=1 baseline) where the biased variant beats baseline. 0.5 = no
      effect; the bias-free measure of benefit.
  (B) coverage -- fraction of the 500-sample probe still emitting a well-formed
      <explanation>.
One line per K (1,2,4,8), all-layers bias. Story: every line stays at/below 0.5
(never helps) and slides down with b, while coverage collapses -- combining
K-repeat with attention forcing only hurts faster.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

SC = Path("exp/results/warmstart/attnbias_scored")
KS = [1, 2, 4, 8]
BS = [0, 1, 2, 4]
N = 500


def load(stem):
    t = pq.read_table(SC / f"{stem}_scored.parquet")
    sid = t.column("sample_idx").to_pylist()
    mse = np.asarray(t.column("mse_warmstart"))
    return {int(s): float(m) for s, m in zip(sid, mse) if m == m}


base = load("grid_k1_b0")
win = {K: [] for K in KS}
cov = {K: [] for K in KS}
for K in KS:
    for b in BS:
        var = load(f"grid_k{K}_b{b}")
        common = sorted(set(base) & set(var))
        bb = np.array([base[s] for s in common]); vv = np.array([var[s] for s in common])
        # self-cell (K1,b0): equal -> report 0.5 (no difference)
        w = 0.5 if (K == 1 and b == 0) else float((vv < bb).mean())
        win[K].append(w)
        cov[K].append(100.0 * len(var) / N)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
colors = {1: "C0", 2: "C2", 4: "C1", 8: "C3"}

axA.axhline(0.5, color="gray", ls="--", lw=1.2, label="no effect (0.5)")
for K in KS:
    axA.plot(BS, win[K], "o-", color=colors[K], lw=2, ms=7, label=f"K={K}")
axA.set_xlabel("attention-logit bias  b  (added to each marker key)")
axA.set_ylabel("paired win-rate vs baseline\n(>0.5 = helps)")
axA.set_title("Forcing attention to the marker(s) never helps")
axA.set_ylim(0, 0.62); axA.set_xticks(BS)
axA.grid(alpha=0.3); axA.legend(fontsize=9, ncol=2)

for K in KS:
    axB.plot(BS, cov[K], "o-", color=colors[K], lw=2, ms=7, label=f"K={K}")
axB.set_xlabel("attention-logit bias  b")
axB.set_ylabel("coverage: well-formed output (%)")
axB.set_title("...and strong bias collapses generation")
axB.set_ylim(60, 102); axB.set_xticks(BS)
axB.grid(alpha=0.3); axB.legend(fontsize=9, ncol=2)

fig.suptitle("Attention-bias x K-repeat (gemma3-12b AV, all layers, n=500)", y=1.02, fontsize=12)
fig.tight_layout()
out = Path("exp/figures/av_attnbias_fve.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
for K in KS:
    print(f"  K={K}: win={[round(w,2) for w in win[K]]}  cov={[round(c) for c in cov[K]]}")
