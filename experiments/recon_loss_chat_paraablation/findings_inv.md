# Findings — inverse paragraph ablation (P3 ablations)

## Question

The first ablation removed paragraphs 1 & 2 (replaced with constants),
keeping the final paragraph — see `findings.md`. The result was that the
final paragraph alone captures almost all of the AR's recoverable signal
(NMSE 0.0087 → 0.0157 on asst_content tokens).

This experiment does the opposite: keep paragraphs 1 & 2 unchanged, and
either:

- **`removed_final`** — drop paragraph 3 entirely (the AR sees only the
  AV's structure / sentence-setup framing, with no per-token continuation).
- **`const_final`** — replace paragraph 3 with a fixed, plausible-looking
  but topically unrelated `Final token "wide"…` sentence. The AV-style
  surface form is preserved but the actual per-token claim is wrong for
  every sample.

## Headline (asst_content tokens, n = 15,297)

| variant         | mean NMSE | median | p10    | p90    | Δ mean   | %worse |
|-----------------|----------:|-------:|-------:|-------:|---------:|-------:|
| original        | 0.0087    | 0.0070 | 0.0039 | 0.0155 | +0.0000  |   0.0% |
| final-¶ only    | 0.0157    | 0.0130 | 0.0081 | 0.0260 | +0.0070  |  98.6% |
| **removed_final** | **0.0196** | 0.0174 | 0.0089 | 0.0318 | +0.0108 |  98.9% |
| **const_final**   | **0.0424** | 0.0402 | 0.0317 | 0.0553 | +0.0337 | **100.0%** |

Two clear results:

1. **Removing the final paragraph is only slightly worse than keeping
   only the final paragraph.** removed_final (0.0196) and final-¶-only
   (0.0157) are both ~2× worse than original (0.0087), but neither
   collapses. The first two paragraphs and the third paragraph each
   carry roughly comparable (and partially overlapping) information about
   the activation — no single paragraph is critical on its own.

2. **A misleading "Final token …" sentence is far worse than no final
   sentence at all.** const_final (0.0424) is **2.2× worse than
   removed_final** and ~5× worse than the original — and *every single
   asst_content sample* gets worse. The AR is not just losing signal;
   it is being actively misled. It commits to the wrong final token
   (`"wide"`) and produces a reconstruction that points away from the
   true activation.

## Per-role breakdown (mean NMSE)

| role          |     n | original | final_only | removed_final | const_final |
|---------------|------:|---------:|-----------:|--------------:|------------:|
| asst_content  | 15,297| 0.0087   | 0.0157     | 0.0196        | 0.0424      |
| user_content  |  4,254| 0.0123   | 0.0196     | 0.0227        | 0.0489      |
| bos           |     27| 0.0006   | 0.0059     | 0.0018        | 0.0188      |
| end_of_turn   |     58| 0.0320   | 0.0368     | 0.0563        | 0.1000      |
| role_marker   |    244| 0.1392   | 0.1224     | 0.1685        | 0.1966      |

The **const_final penalty is uniform across roles** — every role gets
~3-5× worse vs original. That fits the "actively misled" interpretation:
a misleading final-token claim is wrong for *all* token types, not just
content tokens.

`bos` is interesting: removed_final actually leaves it almost as good as
the original (0.0018 vs 0.0006), because for a BOS token there's nothing
useful in P3 anyway — the AV's "Final token …" sentence is most likely
generic. But the const_final paragraph's wrong claim still hurts (0.0188).

## Distribution shape (asst_content)

See `data/nmse_all_variants_dist.png` — the four distributions sit
cleanly in order: original (tight, around 0.005), final-only and
removed_final (overlapping, both around 0.015–0.020), and const_final
shifted way right (around 0.04). The const_final distribution does **not
overlap** the original at all on the bulk: the worst original samples
sit at NMSE ~0.02, exactly where the *median* const_final sample lives.

See `data/nmse_all_variants_bars.png` for a one-glance comparison of
mean NMSE.

## Interpretation

Combined with the previous experiment (`findings.md`), the picture is:

- **Each of the three AV paragraphs carries information** that the AR
  uses, but the information is largely **redundant** between paragraphs.
  Dropping any single paragraph (whether P1+P2 → constants or P3 → none)
  raises NMSE by ~0.007–0.011 — a similar order of magnitude in either
  direction.
- The AR is **not selectively attending only to the final paragraph** —
  if it were, removed_final would be catastrophic and final-only would
  match the original. Neither happens.
- The AR **does take the AV's claim about the final token at face
  value**. A correct-looking but wrong "Final token X…" sentence destroys
  the reconstruction more than removing the sentence entirely. This is a
  meaningful safety property: garbage AV outputs that look right in
  surface form will produce worse downstream activations than garbage
  that looks obviously broken.

Practical implication: when filtering AV outputs for downstream use,
prioritize detecting *plausible-but-wrong* "Final token …" sentences
(e.g. ones whose quoted token doesn't match the actual sampled token).
Those failures are 2× more harmful than missing-sentence failures.

## Files

- `data/comparison_inv.parquet` — 19,880 rows, 605 MB. Joined per-sample
  data with all 4 recon vectors (`recon`, `recon_final_only`,
  `recon_removed_final`, `recon_const_final`) plus NMSE columns and the
  full activation/role/logprob context.
- `data/ar_input_removed_final.parquet`, `data/ar_input_const_final.parquet`
  — the modified explanations fed to the AR (reproduces the inputs).
- `data/ar_output_removed_final.parquet`, `data/ar_output_const_final.parquet`
  — the per-sample modified recons from each variant.
- Plots:
  - `data/nmse_shift_asst_removed_dist.png` / `…_delta.png`
  - `data/nmse_shift_asst_const_dist.png`   / `…_delta.png`
  - `data/nmse_all_variants_dist.png`       — overlay of all 4 variants
  - `data/nmse_all_variants_bars.png`       — mean-NMSE bar chart
- `data/comparison_inv_summary.txt` — text dump of the headline tables.

## Constants used

CONST3 (the misleading P3 for `const_final`):

> *Final token "wide" ends a noun phrase ("if the data has a wide…due to
> a wide…"), requiring a noun phrase like "range of values" or
> "distribution" — likely "range of values" or "unscaled range" or
> "of outliers" or "variance" — describing the problematic input scaling
> issue that causes instability in the model.*

This was deliberately chosen to (a) match the AV's stylistic surface
form, and (b) be topically unrelated to chat content (it talks about
batch-norm instability instead). Specifically the claimed final token
`"wide"` is wrong for ~100% of samples.
