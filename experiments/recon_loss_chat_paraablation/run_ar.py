"""Remote AR-only reconstruction script for the paragraph-ablation experiment.

For each row in the input parquet (sample_idx, modified_explanation):
  - Call critic.reconstruct(modified_explanation) → float32[d_model]
  - Write (sample_idx, modified_recon) to the output parquet

We do NOT compute NMSE / cos here — activations live on the source host.
The post-hoc join + NMSE comparison happens locally after we pull the output.

This script must live alongside `nla_inference.py` (same dir on the pod).

Usage on the pod:
    cd /workspace
    python3 run_ar.py \
        --ar-checkpoint nla-ar \
        --input ar_input.parquet \
        --out ar_output.parquet \
        --ar-device cuda:0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar-checkpoint", required=True)
    ap.add_argument("--input", required=True,
                    help="parquet with (sample_idx, modified_explanation)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ar-device", default="cuda:0")
    ap.add_argument("--flush-every", type=int, default=1024)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from nla_inference import NLACritic

    print(f"[ar] loading NLA critic from {args.ar_checkpoint}", flush=True)
    critic = NLACritic(args.ar_checkpoint, device=args.ar_device)
    d_model = critic.value_head.in_features
    print(f"[ar] d_model={d_model}", flush=True)

    print(f"[ar] reading {args.input}", flush=True)
    tbl = pq.read_table(args.input)
    df = tbl.to_pandas()
    n = len(df)
    print(f"[ar] {n} rows", flush=True)

    out_schema = pa.schema([
        pa.field("sample_idx", pa.int64()),
        pa.field("modified_recon", pa.list_(pa.float32(), d_model)),
    ])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(args.out), out_schema, compression="zstd")

    rows = {"sample_idx": [], "modified_recon": []}

    t0 = time.time()
    last_log = t0
    for i, r in enumerate(df.itertuples(index=False)):
        rec = critic.reconstruct(r.modified_explanation)
        rows["sample_idx"].append(int(r.sample_idx))
        rows["modified_recon"].append(rec.to(torch.float32).cpu().numpy().tolist())

        if (i + 1) % args.flush_every == 0:
            writer.write_table(pa.table(rows, schema=out_schema))
            rows = {"sample_idx": [], "modified_recon": []}
            now = time.time()
            elapsed = now - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n - (i + 1)) / max(rate, 1e-6)
            print(f"[ar] {i+1}/{n}  {rate:.1f}/s  eta={eta/60:.1f}min",
                  flush=True)
            last_log = now

    if rows["sample_idx"]:
        writer.write_table(pa.table(rows, schema=out_schema))
    writer.close()

    elapsed = time.time() - t0
    print(f"[ar] done. {n} rows in {elapsed/60:.2f}min "
          f"({n/elapsed:.1f}/s avg)", flush=True)
    out_sz = Path(args.out).stat().st_size / 1e6
    print(f"[ar] wrote {args.out}  ({out_sz:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
