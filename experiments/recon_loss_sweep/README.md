# Reconstruction-loss sweep experiment

**Goal:** Understand which kinds of tokens have high vs low reconstruction
loss in the NLA round-trip (AV → AR), using the Gemma-3-12B NLA pair on
common-pile/comma_v0.1_training_dataset activations.

`reconstruction loss` = how different an activation is from itself after being
verbalized to text by AV and reconstructed back to a vector by AR. Reported as
both MSE (under √d normalization, range [0, 4]) and cosine similarity.

## Setup

- **Base model:** `google/gemma-3-12b-it`, layer 32 (≈ 2/3 depth of 48).
- **AV (verbalizer):** `kitft/nla-gemma3-12b-L32-av` (~24 GB)
- **AR (reconstructor):** `kitft/nla-gemma3-12b-L32-ar` (~16 GB, truncated to 33 layers)
- **Dataset:** `common-pile/comma_v0.1_training_dataset` (streaming, shuffled with `buffer_size=10_000`).
- **Sampling:** for each text, pick one random token position with
  `position ≥ 10` and that is not a special token / pad. Capture residual-stream
  activation at layer 32.
- **Hardware:** 1× H100 SXM 80GB (RunPod, US-NE-1, $2.99/hr).
  Pod ID: `abfk12qdlxwyvp`.

## Why Gemma-3-12B and not 27B

Gemma-3-27B AV (~54 GB) + AR (~36 GB) cannot fit on one H100. The 12B pair
fits comfortably (AV ~24 GB + AR ~16 GB ≈ 40 GB peak). 27B would have required
2× H100 and ~3× the cost.

## Plan

1. **Dry run, n=20k.** Confirm pipeline works end-to-end, sanity-check a few
   sample explanations, confirm AV parse rate is high and MSE distribution is
   not pathological.
2. **Full sweep, n=100k.** If dry run is clean.
3. **Token-level analysis.** Group results by token, compute mean MSE/cos per
   token, look at the head and tail.

## Phases

The script (`recon_loss_sweep.py`) splits work into two phases so a single H100
isn't asked to hold base + AV + AR simultaneously:

1. **Extract** — load base model only, hook layer 32, write `activations.parquet`
   with columns `(sample_idx, token_id, token_str, position, seq_len, vec_norm,
   text_preview, activation_fp16)`.
2. **Decode** — sglang serves AV, AR runs locally on the same H100. For each
   activation: AV verbalizes → AR reconstructs → MSE & cos vs. original.

## Files in this directory

- `README.md` — this file
- `runbook.md` — exact commands run, in order, with timestamps and outcomes
- `results_dryrun_20k.md` — analysis of the 20k dry-run sweep
- `results_full_100k.md` — analysis of the full 100k sweep
- (data parquets land in `data/` and are git-ignored — too large for the repo)

## Cost guard

The pod is destroyed at the end of the run. If you see a recent commit
mentioning "destroy pod" land here, the pod is already off. If not, check
`curl -H "Authorization: Bearer $(cat ~/.runpod_key)" https://rest.runpod.io/v1/pods`
and DELETE any "RUNNING" pod with the name `nla-recon-loss-sweep`.
