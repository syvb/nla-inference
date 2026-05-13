"""Score the logprob of each sampled token under google/gemma-3-12b-it locally.

For each sample row in results_chat_20k.parquet:
  - Use sample's conv_hash to look up the regenerated conversation
  - Re-tokenize the full chat (same as recon_chat.py extract)
  - Find the sample's position via unpadded_pos (or recompute from chars_before
    as a backup verification)
  - Forward the conversation through Gemma-3-12B
  - At the position just before the sampled token, read log_softmax(logits)
  - Record: target_logprob, top1_token, top1_logprob, target_rank, target_in_topK

Output: parquet with sample_idx + new logprob columns. Joinable to results_chat_20k
by sample_idx.

Strategy:
  - Group samples by conv_hash (most convs have only 1 sampled token; some have
    multiple if duplicates exist — none expected here)
  - For each conv, tokenize once, forward once, extract all needed positions
  - Batch convs by similar length to maximize GPU utilization

Top-K reporting: K=50 by default (gives meaningful rank for "common" tokens).
For tokens not in top-50 we report rank=null but still record the exact
target_logprob (we have the full vocab logits available).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# Imports from the chat extract script for chat-template consistency
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--results-parquet", required=True,
                    help="results_chat_20k.parquet from recon_chat decode")
    ap.add_argument("--regen-jsonl", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Number of convs per forward batch")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from recon_chat import format_and_tokenize_chat

    # cuDNN SDPA issue with Gemma-3 (per recon_loss_sweep notes)
    torch.backends.cuda.enable_cudnn_sdp(False)

    print(f"[score] loading {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True,
    )
    model.eval()

    # Load conv index
    print(f"[score] loading conv index from {args.regen_jsonl}")
    conv_idx: dict[str, list[dict]] = {}
    with Path(args.regen_jsonl).open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ch = rec.get("conv_hash")
            if ch:
                conv_idx[ch] = rec.get("messages") or []
    print(f"[score] {len(conv_idx)} convs")

    # Load samples
    print(f"[score] loading {args.results_parquet}")
    samples = pq.read_table(args.results_parquet, columns=[
        "sample_idx", "conv_hash", "token_id", "token_str",
        "position", "unpadded_pos", "role", "seq_len",
    ]).to_pandas()
    n_samples = len(samples)
    print(f"[score] {n_samples} samples")

    # Group samples by conv_hash. (Each row has unpadded_pos in the
    # tokenized conv. We re-tokenize, then index that position.)
    by_conv: dict[str, list[int]] = defaultdict(list)
    for i, r in samples.iterrows():
        by_conv[r.conv_hash].append(i)

    n_convs = len(by_conv)
    print(f"[score] {n_convs} unique convs to process")

    # Storage
    target_logprob = np.full(n_samples, np.nan, dtype=np.float32)
    top1_token = [""] * n_samples
    top1_logprob = np.full(n_samples, np.nan, dtype=np.float32)
    target_rank = np.full(n_samples, -1, dtype=np.int32)
    in_topk = np.zeros(n_samples, dtype=bool)
    error_flags = [""] * n_samples

    BOS_ID = tok.bos_token_id or 2
    PAD_ID = tok.pad_token_id

    t0 = time.time()
    convs_done = 0

    # Process one conv at a time (simpler, batching adds complexity for variable len)
    # On H100 single forward at seq=2048 is fast enough.
    with torch.no_grad():
        for conv_hash, sample_idxs in by_conv.items():
            convs_done += 1
            messages = conv_idx.get(conv_hash)
            if not messages:
                for si in sample_idxs:
                    error_flags[si] = "no_conv"
                continue
            try:
                ids, _roles = format_and_tokenize_chat(messages, tok, args.max_len)
            except Exception as e:
                for si in sample_idxs:
                    error_flags[si] = f"tok:{type(e).__name__}"
                continue
            if len(ids) < 2:
                for si in sample_idxs:
                    error_flags[si] = "too_short"
                continue

            # Forward
            input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
            try:
                out = model(input_ids=input_ids, use_cache=False)
            except Exception as e:
                for si in sample_idxs:
                    error_flags[si] = f"fwd:{type(e).__name__}"
                continue
            logits = out.logits[0]  # [T, vocab]
            # logits[t] predicts ids[t+1]; so logprob of ids[t] given prefix is
            # log_softmax(logits[t-1])[ids[t]].
            log_probs = F.log_softmax(logits.float(), dim=-1)
            del out, logits

            for si in sample_idxs:
                r = samples.iloc[si]
                pos = int(r.unpadded_pos)
                if pos < 1 or pos >= len(ids):
                    error_flags[si] = f"pos_oob({pos}/{len(ids)})"
                    continue
                # The token we want: ids[pos], conditioned on ids[:pos]
                # We use logits at position pos-1.
                lp_row = log_probs[pos - 1]  # [vocab]
                target_tok_id = int(r.token_id)
                # Sanity: should match what's at ids[pos]
                actual_tok_id = ids[pos]
                if actual_tok_id != target_tok_id:
                    # Tokenization drift — record but still try
                    error_flags[si] = f"tok_drift({actual_tok_id}!={target_tok_id})"
                    # Use the recorded target_tok_id for logprob lookup
                target_logprob[si] = float(lp_row[target_tok_id].item())
                # Top-K
                topk = torch.topk(lp_row, k=args.top_k)
                topk_ids = topk.indices.cpu().tolist()
                topk_lps = topk.values.cpu().tolist()
                top1_token[si] = tok.decode([topk_ids[0]])
                top1_logprob[si] = float(topk_lps[0])
                if target_tok_id in topk_ids:
                    target_rank[si] = topk_ids.index(target_tok_id) + 1
                    in_topk[si] = True

            if convs_done % 200 == 0:
                elapsed = time.time() - t0
                rate = convs_done / max(elapsed, 1e-6)
                eta = (n_convs - convs_done) / max(rate, 1e-6)
                print(f"[score] {convs_done}/{n_convs}  {rate:.1f}c/s  eta={eta/60:.1f}min",
                      flush=True)

    print(f"[score] forward pass done. {convs_done} convs in {(time.time()-t0)/60:.1f}min")

    # Build output
    samples["target_logprob"] = target_logprob
    samples["top1_token"] = top1_token
    samples["top1_logprob"] = top1_logprob
    samples["target_rank"] = target_rank
    samples["in_topk"] = in_topk
    samples["error"] = error_flags

    n_ok = (samples["error"] == "").sum()
    n_in_topk = samples["in_topk"].sum()
    print(f"[score] succeeded: {n_ok}/{n_samples}  ({100*n_ok/n_samples:.1f}%)")
    print(f"[score] in_top{args.top_k}: {n_in_topk}/{n_samples}  "
          f"({100*n_in_topk/n_samples:.1f}%)")

    # Save (only the new cols + sample_idx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = ["sample_idx", "conv_hash", "token_id", "token_str", "role",
            "position", "unpadded_pos",
            "target_logprob", "top1_token", "top1_logprob",
            "target_rank", "in_topk", "error"]
    pq.write_table(pa.Table.from_pandas(samples[keep], preserve_index=False),
                   out_path, compression="zstd")
    print(f"[score] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
