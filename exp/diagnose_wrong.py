"""Diagnose why wrong-token reconstruction is almost as good as right-token.

H1: random permutation often maps a row to the same token (chat has lots of
    repeated common tokens).
H2: even when the token is genuinely different, the AR doesn't care much —
    most of its signal is the prompt scaffold.
"""
import json
import numpy as np
import pyarrow.parquet as pq
from collections import Counter

t = pq.read_table("/home/debian/nla-inference/exp/results/controls_full.parquet")
token = t.column("token_str").to_pylist()
swap = t.column("swap_mse_nrm").to_numpy()
wrong = t.column("wrong_token_mse_nrm").to_numpy()
empty = t.column("empty_mse_nrm").to_numpy()
base = t.column("base_mse_nrm").to_numpy()

# Recover the wrong-token mapping from the saved swap text vs wrong_token_text.
wrong_text = t.column("wrong_token_text").to_pylist()
# Pattern: `Final token: <json-escaped token>`. Decode the JSON to get the token back.
def extract_token(s):
    return json.loads(s.split("Final token: ", 1)[1])

wrong_tokens = [extract_token(s) for s in wrong_text]
n = len(token)

# H1: collision rate
collisions = sum(1 for a, b in zip(token, wrong_tokens) if a == b)
print(f"wrong-token equals real token by coincidence: {collisions}/{n} ({100*collisions/n:.1f}%)")

# Token frequency
freq = Counter(token)
top = freq.most_common(10)
print(f"\ntop-10 most common tokens in dataset:")
for t_, c in top:
    print(f"  {c:5d}  {t_!r}")

# Average mse_nrm split by whether wrong-token collided
coll = np.array([a == b for a, b in zip(token, wrong_tokens)])
print(f"\n                    n     swap     wrong    empty    base")
for label, mask in [("collided   ", coll), ("not-collided", ~coll)]:
    if mask.sum():
        print(f"  {label}    {mask.sum():5d}  {swap[mask].mean():.4f}  "
              f"{wrong[mask].mean():.4f}  {empty[mask].mean():.4f}  {base[mask].mean():.4f}")

# Bin by "how different is the wrong token from the right one in popularity":
# if wrong token is also extremely common, may reconstruct just as well.
print("\nWrong-token mse by token-popularity bucket of the WRONG token:")
buckets = []
for tk, w in zip(wrong_tokens, wrong):
    f = freq.get(tk, 0)
    buckets.append((f, w))
buckets = np.array(buckets)
freqs = buckets[:, 0]
mses = buckets[:, 1]
# Quintile buckets
qs = np.percentile(freqs, [20, 40, 60, 80])
print(f"  quintile boundaries (freq): {qs}")
print(f"  {'bucket':25s}  n     mean_mse")
prev = 0
for i, q in enumerate(list(qs) + [freqs.max() + 1]):
    mask = (freqs > prev) & (freqs <= q)
    if mask.sum():
        print(f"  freq in ({prev:.0f},{q:.0f}]:        {mask.sum():5d}  {mses[mask].mean():.4f}")
    prev = q
