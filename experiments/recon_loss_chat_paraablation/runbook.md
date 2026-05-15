# Runbook — paragraph ablation (recon_loss_chat_paraablation)

## Question

How much of the reconstruction quality of NLA on chat data comes from
the *final paragraph* of the AV explanation alone? The AV typically emits
3 paragraphs:

1. structure framing ("Technical Q&A forum format…")
2. context-setup sentence ("The phrase 'X' signals…")
3. final-token continuation ("Final token 'Y' opens a clause requiring…")

If most of the recon signal lives in #3, then replacing #1 and #2 with
*fixed constants* should leave NMSE roughly unchanged. If #1/#2 carry
real information, NMSE will get noticeably worse.

## Setup

- Source dataset: `experiments/recon_loss_chat/data/results_chat_20k.parquet`
  (19,881 chat samples, Gemma-3-12B-it activations at L32 reconstructed
  via the chat dataset built earlier).
- Filter: `av_parsed=True` AND non-empty `recon` → 19,880 rows kept.
- Modified explanation construction (per row):
  - `final_para = explanation.split("\n")[-1]` (last newline-split chunk;
    walks back to non-empty if trailing newline)
  - `modified = CONST1 + "\n\n" + CONST2 + "\n\n" + final_para`
- Constants (chosen by user, unrelated to chat content):
  - CONST1: *"Structured ML/data science explanation format: structured
    advice with code blocks and conceptual framing establishes a technical
    troubleshooting guide."*
  - CONST2: *"The sentence \"If your data has a wide\" sets up a problem
    statement about batch normalization instability, specifically the issue
    of feature scaling or a large input range."*

## Pipeline

1. **Local prep** (`prepare_subset.py`):
   - Reads `results_chat_20k.parquet`, filters, builds modified text.
   - Writes `data/ar_input.parquet` with (sample_idx, modified_explanation).
   - Size: 2.44 MB / 19,880 rows.
2. **Remote AR** (H100 pod, `run_ar.py`):
   - Loads only `syvb/nla-ar` checkpoint (no base LM, no AV / sglang).
   - Calls `NLACritic.reconstruct(modified)` per row.
   - Writes `ar_output.parquet` with (sample_idx, modified_recon).
3. **Local post-processing**:
   - Join modified_recon back into the main results parquet.
   - NMSE = 2·(1 − cos(modified_recon, activation)).
   - Compare side-by-side to existing NMSE from original recon.

## Pod

- Created: 2026-05-14 22:53 UTC.
- Pod ID `q1h018uhl0x9p7`, machine `774gk4vddbyz`, AP-IN-1, $2.99/hr.
- SSH `103.207.149.145:18317`.
- Container disk 80 GB (only nla-ar ~16 GB needed).

## Timeline

- 22:53 UTC — pod creation
- 22:56 UTC — IP/port assigned (~3 min wait)
- 22:57 UTC — files scp'd, install kicked off
- 22:59 UTC — install done (`python -m pip` of transformers 5.6+, etc.)
- 23:00 UTC — first download attempt failed (wrong repo name `syvb/nla-ar`)
- 23:00–23:02 UTC — download of `kitft/nla-gemma3-12b-L32-ar` (16 GB)
- 23:02 UTC — first AR run failed: `ModuleNotFoundError: numpy` because
  `python3` is system-Python (3.10, bare) while `python` is the pytorch
  image's preinstalled env (3.12, with torch + numpy). Patched
  `pod_runner.sh` to use `python` for both pip and run invocations.
- 23:02–23:10 UTC — AR run: 19 880 rows / 8.3 min / **39.9/s avg** on a
  single H100 (serial, no batching).
- 23:11 UTC — output (115 MB) scp'd back.
- 23:12 UTC — pod DELETE → HTTP 204; verify 404.

**Pod cost: ~18 min × $2.99/hr ≈ $0.90.**

## Result

See `findings.md`. Headline: mean NMSE 0.0111 → 0.0179 (+61 % rel),
98.2 % of samples worse, but absolute degradation small — the final
paragraph alone captures ~99.7 % of the variance-reduction the full
explanation provides. Only `role_marker` tokens (n=244) actually
improve under the ablation; all four other roles get uniformly worse
by Δ NMSE ≈ +0.005…+0.007.

## Run 2 — inverse ablation (2026-05-14, ~23:35 UTC)

Same source dataset, but ablating P3 instead of P1+P2:

- `removed_final`: explanation = P1+P2 (drop P3 entirely)
- `const_final`:   explanation = P1+P2 + canned "Final token 'wide'…"

Both run in one pod via the new `pod_runner.sh run-many` command.

- 23:30 UTC — pod creation (`sh9ftnr17hh7of`, AP-IN-1, $2.99/hr)
- 23:33 UTC — IP/port assigned (~3 min wait)
- 23:35 UTC — files scp'd
- 23:35–23:38 UTC — install + AR download (cached image, fast)
- 23:38–23:46 UTC — AR variant 1 (removed_final): 19,880 rows / 8.95 min / 37.0/s
- 23:46–23:55 UTC — AR variant 2 (const_final):   19,880 rows / 8.88 min / 37.3/s
- 23:57 UTC — outputs scp'd back
- 23:58 UTC — pod DELETE → 204 / 404

**Pod cost: ~28 min × $2.99/hr ≈ $1.40.**

### Inverse-result headlines (asst_content)

| variant            | mean NMSE | Δ vs orig | %worse |
|--------------------|----------:|----------:|-------:|
| original           | 0.0087    | +0.0000   |   0.0% |
| final-¶ only       | 0.0157    | +0.0070   |  98.6% |
| removed_final      | 0.0196    | +0.0108   |  98.9% |
| const_final        | **0.0424**| **+0.0337** | **100.0%** |

See `findings_inv.md` for details. The headline finding: a *misleading*
canned final paragraph is ~2× worse than no final paragraph, and ~5×
worse than the original — the AR commits to the wrong target token and
produces an actively-wrong reconstruction.

## Tooling notes for next time

- The RunPod pytorch image has two Python interpreters: `python`
  (`/usr/local/bin/python`, 3.12, has torch/numpy preinstalled) and
  `python3` (`/usr/bin/python3`, 3.10, bare). Always use `python` and
  `python -m pip` on this image.
- `kitft/nla-gemma3-12b-L32-ar` is the right AR repo name for Gemma-3.
  See `experiments/recon_loss_sweep/pod_runner.sh` for the canonical list
  of NLA repos.
- AR-only at single-row serial is ~40/s on H100 — fine for 20k. If
  batching were needed for >100k, would have to extend
  `NLACritic.reconstruct` to accept a list.
