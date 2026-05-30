"""Recompute every FINDINGS FVE table using the paper aggregate formula with the
mean-activation baseline:  FVE = 1 - mean(mse_variant) / mean(mse_to_mean),
where mse_to_mean = mse_nrm(dataset_mean_activation, sample_activation).

All mse are the direction-only mse_nrm already stored in the out parquets, so
the denominator uses the same metric (mse_nrm to the mean activation).
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

MSE_SCALE = 61.96773353931867
CHAT_ACT = "exp/results/warmstart/activations_chat_20k.parquet"
CHAT_RES = "exp/results/warmstart/results_chat_20k.parquet"
PT_ACT = "/home/debian/.cache/huggingface/hub/datasets--syvb--nla-recon-loss-sweep/snapshots/47cc32541a3a6eaac63c48916bb8f61a38e5dd68/data/activations_gemma12_diverse_shards_seed0_20000.parquet"
PT_RES = "/home/debian/.cache/huggingface/hub/datasets--syvb--nla-recon-loss-sweep/snapshots/47cc32541a3a6eaac63c48916bb8f61a38e5dd68/data/results_gemma12_diverse_shards_seed0_20000.parquet"
D = 3840


def mse_nrm_batch(preds, golds, s=MSE_SCALE):
    pn = preds / (np.linalg.norm(preds, axis=-1, keepdims=True) + 1e-12) * s
    gn = golds / (np.linalg.norm(golds, axis=-1, keepdims=True) + 1e-12) * s
    return ((pn - gn) ** 2).mean(axis=-1)


def load_acts(path, parsed_only_via=None):
    """Return (sample_idx_array, activation_matrix). If parsed_only_via given
    (a results parquet path), mean is taken over av_parsed rows only — but we
    always return ALL activations for per-sample lookup."""
    t = pq.read_table(path, columns=["sample_idx", "activation"])
    n = t.num_rows
    sids = np.asarray(t.column("sample_idx").to_pylist())
    acts = np.asarray(t.column("activation").combine_chunks().values).astype(np.float32).reshape(n, -1)
    return sids, acts


def mean_activation(res_path, act_sids, acts):
    """Mean activation over av_parsed rows (matches the established 0.677/0.723)."""
    r = pq.read_table(res_path, columns=["sample_idx", "av_parsed"])
    rp = {r.column("sample_idx")[i].as_py(): r.column("av_parsed")[i].as_py() for i in range(r.num_rows)}
    pos = {int(s): i for i, s in enumerate(act_sids)}
    keep = [pos[s] for s in rp if rp[s] and s in pos]
    return acts[keep].mean(axis=0)


def build_to_mean(act_sids, acts, mean_act):
    """dict sample_idx -> mse_nrm(mean_act, sample_act)."""
    tm = mse_nrm_batch(np.broadcast_to(mean_act, acts.shape), acts)
    return {int(s): tm[i] for i, s in enumerate(act_sids)}


def fve_agg(mse_variant, to_mean_vals):
    """Paper aggregate FVE on the given (already-row-aligned) arrays."""
    return 1 - mse_variant.mean() / to_mean_vals.mean()


# ---- load activations + per-sample to-mean denominators ----
print("loading chat activations...")
c_sids, c_acts = load_acts(CHAT_ACT)
c_mean = mean_activation(CHAT_RES, c_sids, c_acts)
c_tm = build_to_mean(c_sids, c_acts, c_mean)
print(f"  chat mean mse_to_mean = {np.mean(list(c_tm.values())):.4f}")

print("loading PT activations...")
p_sids, p_acts = load_acts(PT_ACT)
p_mean = mean_activation(PT_RES, p_sids, p_acts)
p_tm = build_to_mean(p_sids, p_acts, p_mean)
print(f"  PT mean mse_to_mean = {np.mean(list(p_tm.values())):.4f}")


def variant_fve(out_path, mse_col, tm, parsed_col="warmstart_parsed", extra_mask_sids=None):
    t = pq.read_table(out_path)
    sids = t.column("sample_idx").to_pylist()
    mse = np.asarray(t.column(mse_col).to_pylist(), dtype=np.float64)
    parsed = np.asarray(t.column(parsed_col).to_pylist()) if parsed_col in t.column_names else np.ones(len(sids), bool)
    rows = []
    for i, sid in enumerate(sids):
        if not parsed[i] or np.isnan(mse[i]): continue
        if extra_mask_sids is not None and sid not in extra_mask_sids: continue
        if sid not in tm: continue
        rows.append((mse[i], tm[sid]))
    rows = np.array(rows)
    return fve_agg(rows[:, 0], rows[:, 1]), len(rows)


def shared_parsed(out_paths, tm):
    """sample_idx parsed in all out_paths (warmstart_parsed) and in tm."""
    sets = []
    for p in out_paths:
        t = pq.read_table(p, columns=["sample_idx", "warmstart_parsed"])
        s = {t.column("sample_idx")[i].as_py() for i in range(t.num_rows)
             if t.column("warmstart_parsed")[i].as_py()}
        sets.append(s)
    sh = set.intersection(*sets)
    return {s for s in sh if s in tm}


# ============ TABLE 1: Thread A chat (§3) ============
print("\n=== §3 Thread A chat ===")
# v2/v3/v4/AV on shared 484 rows of v2/v3/v4 out parquets
v234 = [f"exp/results/warmstart/warmstart_500_v{n}_out.parquet" for n in (2, 3, 4)]
sh234 = shared_parsed(v234, c_tm)
# also require av_parsed in v2
t2 = pq.read_table(v234[0])
i2 = {t2.column("sample_idx")[i].as_py(): i for i in range(t2.num_rows)}
sh234 = {s for s in sh234 if t2.column("av_parsed")[i2[s]].as_py()}
def col_fve(path, col, sids):
    t = pq.read_table(path); idx = {t.column("sample_idx")[i].as_py(): i for i in range(t.num_rows)}
    mse = np.array([t.column(col)[idx[s]].as_py() for s in sids])
    tm = np.array([c_tm[s] for s in sids])
    return fve_agg(mse, tm)
print(f"  n_shared = {len(sh234)}")
print(f"  AV  : {col_fve(v234[0], 'mse_av_rerun', sh234):+.4f}")
print(f"  v2  : {col_fve(v234[0], 'mse_warmstart', sh234):+.4f}")
print(f"  v3  : {col_fve(v234[1], 'mse_warmstart', sh234):+.4f}")
print(f"  v4  : {col_fve(v234[2], 'mse_warmstart', sh234):+.4f}")
# v1 from its own parsed rows
f, n = variant_fve("exp/results/warmstart/warmstart_500_out.parquet", "mse_warmstart", c_tm)
print(f"  v1 (own {n} rows): {f:+.4f}")


# ============ TABLE 2: Thread B PT (§4) ============
print("\n=== §4 Thread B PT (shared rows) ===")
pt_variants = {
    "AV (baseline)":   ("exp/results/warmstart/warmstart_pt_v7_user_out.parquet", "mse_av_rerun"),
    "input_only":      ("exp/results/warmstart/warmstart_pt_input_only_out.parquet", "mse_warmstart"),
    "v7_swap":         ("exp/results/warmstart/warmstart_pt_v7_swap_out.parquet", "mse_warmstart"),
    "v7delim":         ("exp/results/warmstart/warmstart_pt_v7delim_out.parquet", "mse_warmstart"),
    "av_delim_swap":   ("exp/results/warmstart/warmstart_pt_av_delim_swap_out.parquet", "mse_warmstart"),
    "av_delim":        ("exp/results/warmstart/warmstart_pt_av_delim_out.parquet", "mse_warmstart"),
}
sh_pt = shared_parsed([p for p, _ in pt_variants.values()], p_tm)
print(f"  n_shared = {len(sh_pt)}")
for name, (path, col) in pt_variants.items():
    t = pq.read_table(path); idx = {t.column("sample_idx")[i].as_py(): i for i in range(t.num_rows)}
    mse = np.array([t.column(col)[idx[s]].as_py() for s in sh_pt])
    tm = np.array([p_tm[s] for s in sh_pt])
    print(f"  {name:16s}: {fve_agg(mse, tm):+.4f}")


# ============ §4 chat scaling (full 20k) ============
print("\n=== §4 chat av_delim 20k (full) ===")
t = pq.read_table("exp/results/warmstart/warmstart_chat_av_delim_20k_out.parquet")
idx = {t.column("sample_idx")[i].as_py(): i for i in range(t.num_rows)}
# AV baseline from results recon
r = pq.read_table(CHAT_RES, columns=["sample_idx", "activation", "recon", "av_parsed"])
nr = r.num_rows
racts = np.asarray(r.column("activation").combine_chunks().values).astype(np.float32).reshape(nr, -1)
rrec = np.asarray(r.column("recon").combine_chunks().values).astype(np.float32).reshape(nr, -1)
rparsed = np.asarray(r.column("av_parsed").to_pylist())
av_mse_all = mse_nrm_batch(rrec, racts)
ridx = {r.column("sample_idx")[i].as_py(): i for i in range(nr)}
rows_av, rows_avd, rows_tm = [], [], []
for i in range(t.num_rows):
    sid = t.column("sample_idx")[i].as_py()
    if not t.column("warmstart_parsed")[i].as_py(): continue
    if sid not in ridx or not rparsed[ridx[sid]]: continue
    rows_av.append(av_mse_all[ridx[sid]]); rows_avd.append(t.column("mse_warmstart")[i].as_py()); rows_tm.append(c_tm[sid])
rows_av, rows_avd, rows_tm = map(np.array, (rows_av, rows_avd, rows_tm))
print(f"  n = {len(rows_av)}")
print(f"  AV baseline : {fve_agg(rows_av, rows_tm):+.4f}")
print(f"  av_delim    : {fve_agg(rows_avd, rows_tm):+.4f}")


# ============ TABLE 4: Thread C PT (§5) ============
print("\n=== §5 Thread C PT (AV with/without source in prompt) ===")
sh_c = shared_parsed(["exp/results/warmstart/warmstart_pt_av_canonical_full_out.parquet",
                      "exp/results/warmstart/warmstart_pt_av_source_full_out.parquet"], p_tm)
print(f"  n_shared = {len(sh_c)}")
for name, path in [("canonical", "warmstart_pt_av_canonical_full_out"),
                   ("with source", "warmstart_pt_av_source_full_out")]:
    t = pq.read_table(f"exp/results/warmstart/{path}.parquet")
    idx = {t.column("sample_idx")[i].as_py(): i for i in range(t.num_rows)}
    mse = np.array([t.column("mse_warmstart")[idx[s]].as_py() for s in sh_c])
    tm = np.array([p_tm[s] for s in sh_c])
    print(f"  {name:12s}: {fve_agg(mse, tm):+.4f}")


# ============ TABLE 5: §5b chat (AV as text explainer) ============
print("\n=== §5b chat ===")
f_act, n = variant_fve("exp/results/warmstart/warmstart_av_on_text_5k_out.parquet", "mse_av_saved", c_tm)
f_txt, _ = variant_fve("exp/results/warmstart/warmstart_av_on_text_5k_out.parquet", "mse_warmstart", c_tm)
print(f"  AV on activation (n={n}): {f_act:+.4f}")
print(f"  AV on text             : {f_txt:+.4f}")
# Sonnet ref: chat v2 on its own parsed rows
f_son, ns = variant_fve("exp/results/warmstart/warmstart_500_v2_out.parquet", "mse_warmstart", c_tm)
print(f"  Sonnet v2 (ref, n={ns}) : {f_son:+.4f}")
# v4format vs native on shared rows
sh_b = shared_parsed(["exp/results/warmstart/warmstart_av_v4format_5k_out.parquet",
                      "exp/results/warmstart/warmstart_av_on_text_5k_out.parquet"], c_tm)
for name, path in [("v4format", "warmstart_av_v4format_5k_out"),
                   ("native", "warmstart_av_on_text_5k_out")]:
    t = pq.read_table(f"exp/results/warmstart/{path}.parquet")
    idx = {t.column("sample_idx")[i].as_py(): i for i in range(t.num_rows)}
    mse = np.array([t.column("mse_warmstart")[idx[s]].as_py() for s in sh_b])
    tm = np.array([c_tm[s] for s in sh_b])
    print(f"  {name:10s} (shared n={len(sh_b)}): {fve_agg(mse, tm):+.4f}")
