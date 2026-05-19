"""Analyse controls_full.parquet: report distributions + per-row decomposition."""
import pyarrow.parquet as pq
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/debian/nla-inference/exp/results/controls_full.parquet"
t = pq.read_table(path)
base = t.column("base_mse_nrm").to_numpy()
swap = t.column("swap_mse_nrm").to_numpy()
wrong = t.column("wrong_token_mse_nrm").to_numpy()
perm = t.column("permuted_mse_nrm").to_numpy()
empty = t.column("empty_mse_nrm").to_numpy()
n = len(base)
print(f"n = {n}\n")
print(f"{'variant':40s}  mean    median  std     p10     p90")
for name, m in [
    ("baseline (full explanation)", base),
    ("swap (Final token: <real>)", swap),
    ("wrong-token (Final token: <wrong>)", wrong),
    ("permuted (other row's explanation)", perm),
    ("empty ('')", empty),
]:
    print(
        f"  {name:38s} {m.mean():.4f}  {np.median(m):.4f}  {m.std():.4f}  "
        f"{np.percentile(m, 10):.4f}  {np.percentile(m, 90):.4f}"
    )

print("\nFVE over the AR's empty-prompt prior (empty = 0 by construction):")
print(f"{'variant':40s}  mean    median")
for name, m in [
    ("baseline (full explanation)", base),
    ("swap (Final token: <real>)", swap),
    ("wrong-token (Final token: <wrong>)", wrong),
    ("permuted (other expl.)", perm),
]:
    f = (empty - m) / empty
    print(f"  {name:38s} {f.mean():+.4f} {np.median(f):+.4f}")

print()
print("Per-row: fraction of (baseline - empty) MSE-gap recovered by each variant.")
gap = empty - base
mask = gap > 0
print(
    f"  rows where full explanation beats empty: {mask.sum()}/{n} ({100*mask.mean():.1f}%)"
)
for name, m in [
    ("swap (Final token: <real>)", swap),
    ("wrong-token", wrong),
    ("permuted", perm),
]:
    recovered = empty - m
    frac = recovered[mask] / gap[mask]
    print(
        f"  {name:30s}: mean={frac.mean():+.4f} median={np.median(frac):+.4f} "
        f"frac>0={100*(frac>0).mean():.1f}%"
    )

print()
print(
    f"swap < wrong_token (real token helps over wrong token): "
    f"{100*(swap < wrong).mean():.1f}% of rows"
)
print(
    f"swap < empty       (real token helps over empty prompt): "
    f"{100*(swap < empty).mean():.1f}% of rows"
)
print(
    f"swap < baseline    (real token alone matches full expl):  "
    f"{100*(swap < base).mean():.1f}% of rows"
)
