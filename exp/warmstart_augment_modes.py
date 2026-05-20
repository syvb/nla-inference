"""Build augmented warm-start parquets in 4 modes for ablation.

Modes (each produces a separate output parquet):
  v7_swap        — "Analysis:\\n<v7>\\n\\nOriginal text:\\n<input>"      (swap order vs v7delim)
  input_only     — "<input>"                                              (raw input, no analysis)
  av_delim       — "Original text:\\n<input>\\n\\nAnalysis:\\n<av>"       (use actual AV's explanation, input first)
  av_delim_swap  — "Analysis:\\n<av>\\n\\nOriginal text:\\n<input>"       (use actual AV's explanation, analysis first)

The schema mirrors the regular sonnet parquet so the existing AR runner can
consume each output unchanged.
"""
from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonnet", required=True, help="v7 sonnet parquet (for v7 expl)")
    ap.add_argument("--prompts", required=True, help="prompts parquet (decoded_full)")
    ap.add_argument("--results", required=True, help="PT results parquet (av explanation)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    s = pq.read_table(args.sonnet)
    p = pq.read_table(args.prompts)
    r = pq.read_table(args.results, columns=["sample_idx", "explanation", "av_parsed"])

    p_idx = {p.column("sample_idx")[i].as_py(): i for i in range(p.num_rows)}
    r_idx = {r.column("sample_idx")[i].as_py(): i for i in range(r.num_rows)}

    sids = s.column("sample_idx").to_pylist()
    v7_cleaned = s.column("warmstart_cleaned").to_pylist()
    v7_raw = s.column("warmstart_raw").to_pylist()
    v7_parsed = s.column("warmstart_parsed").to_pylist()

    modes = {
        "v7_swap":       [],
        "input_only":    [],
        "av_delim":      [],
        "av_delim_swap": [],
    }
    parsed_for_mode = {k: [] for k in modes}

    for i in range(len(sids)):
        sid = sids[i]
        original = p.column("decoded_full")[p_idx[sid]].as_py() if sid in p_idx else None
        av_expl = None
        av_ok = False
        if sid in r_idx:
            j = r_idx[sid]
            av_ok = bool(r.column("av_parsed")[j].as_py())
            if av_ok:
                av_expl = r.column("explanation")[j].as_py()

        # v7_swap: needs v7_parsed AND original text
        if v7_parsed[i] and v7_cleaned[i] and original is not None:
            modes["v7_swap"].append(f"Analysis:\n{v7_cleaned[i]}\n\nOriginal text:\n{original}")
            parsed_for_mode["v7_swap"].append(True)
        else:
            modes["v7_swap"].append("")
            parsed_for_mode["v7_swap"].append(False)

        # input_only: just raw input
        if original is not None:
            modes["input_only"].append(original)
            parsed_for_mode["input_only"].append(True)
        else:
            modes["input_only"].append("")
            parsed_for_mode["input_only"].append(False)

        # av_delim: original text first then av explanation
        if av_ok and av_expl and original is not None:
            modes["av_delim"].append(f"Original text:\n{original}\n\nAnalysis:\n{av_expl}")
            parsed_for_mode["av_delim"].append(True)
        else:
            modes["av_delim"].append("")
            parsed_for_mode["av_delim"].append(False)

        # av_delim_swap: av explanation first then original text
        if av_ok and av_expl and original is not None:
            modes["av_delim_swap"].append(f"Analysis:\n{av_expl}\n\nOriginal text:\n{original}")
            parsed_for_mode["av_delim_swap"].append(True)
        else:
            modes["av_delim_swap"].append("")
            parsed_for_mode["av_delim_swap"].append(False)

    for name, texts in modes.items():
        out_path = f"{args.out_dir}/sonnet46_pt_{name}.parquet"
        tbl = pa.table({
            "sample_idx": pa.array(sids, type=pa.int64()),
            "warmstart_raw": pa.array(v7_raw),  # original raw, untouched (just for schema completeness)
            "warmstart_cleaned": pa.array(texts),
            "warmstart_parsed": pa.array(parsed_for_mode[name], type=pa.bool_()),
        })
        pq.write_table(tbl, out_path, compression="zstd")
        n_ok = sum(parsed_for_mode[name])
        sample = next((t for j, t in enumerate(texts) if parsed_for_mode[name][j]), "")
        print(f"[{name:14s}] {n_ok}/{len(texts)} valid; first sample len={len(sample)} chars")
        if sample:
            print(f"  head: {sample[:200]!r}")
            print(f"  tail: ...{sample[-200:]!r}")
        print()


if __name__ == "__main__":
    main()
