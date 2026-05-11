# Deep findings: which tokens reconstruct poorly, and why

Synthesis of `deep_analysis_20k.md` (auto-generated tables) over the 20k
Gemma-3-12B / L32 sweep. The earlier `findings.md` covered the headline
distribution; this document drills into the failure modes.

## TL;DR (one screen)

1. Reconstruction is uniformly excellent — mean cos = 0.993, p10 = 0.988,
   p99 = 0.998. **There is no broad axis along which reconstruction is
   "moderately worse"** — instead, ~99.83 % of samples land in a tight
   high-cos band, and the remainder fail catastrophically.
2. Catastrophic failures (cos < 0.7) are **0.17 %** of non-sink samples, and
   they cluster overwhelmingly in *one* cell:
   sentence-initial capitalised words at low L2-norm (≈ 36k).
3. The failure mechanism is **AV template collapse**. When the AV cannot
   extract content from the activation, it emits one of two memorised
   templates verbatim. One is the "high-norm sink" template
   ("Fragmented, incoherent academic text pattern…") which round-trips at
   cos = 1.000 and is a *correct* but information-free reconstruction.
   The other — "Technical article structure: a blog post reviewing a
   specific research paper…" — is the **failure** template; samples that
   trigger it average cos = 0.24.
4. **Token surface form predicts failure better than token frequency or
   position.** Specifically: leading-space (BPE word-initial) tokens
   reconstruct ~2 percentage points better than no-leading-space pieces,
   and within no-leading-space tokens, capitalised-initial words are the
   single weakest cell.

## 1. The category landscape

Mean cos across 8 disjoint token categories (non-sink):

| category                   | n    | cos_mean | cos_p10 | mean ‖v‖ |
|----------------------------|------|----------|---------|----------|
| `cap_no_space`             | 1054 | **0.976** | 0.986   | 68 657   |
| `digit_only`               |  954 | 0.990    | 0.983   | 57 960   |
| `whitespace`               | 1693 | 0.990    | 0.982   | 63 041   |
| `subword_continuation`     | 1888 | 0.991    | 0.986   | 65 033   |
| `punct_only`               | 4289 | 0.992    | 0.986   | 67 904   |
| `single_char_alpha`        | 1165 | 0.993    | 0.988   | 72 295   |
| `cap_with_space`           |  685 | 0.995    | 0.992   | 71 891   |
| `word_lower_with_space`    | 8236 | **0.996** | 0.995   | 80 607   |

Only `cap_no_space` stands meaningfully apart at 0.976 — and even there
the median is 0.995, so the population is bimodal: most samples reconstruct
fine and a small subset fails very hard.

## 2. The case × leading-space cross-cut

Within the alphabetic tokens (the cleanest comparison), the mean-cos
ordering is:

| | n | cos_mean | cos_p10 |
|---|---|---|---|
| `lead-space lower` (typical content word, e.g. ` been`)  | 8553 | 0.996 | 0.994 |
| `lead-space Cap`   (mid-sentence capitalised, e.g. ` Java`) |  942 | 0.995 | 0.992 |
| `no-space lower`   (mid-BPE piece, e.g. `ing`)         | 2266 | 0.991 | 0.986 |
| `no-space Cap`     (sentence-initial cap, e.g. `How`)   | 1267 | **0.978** | 0.986 |

The leading-space (i.e. BPE word-initial) tokens beat their no-space
counterparts by 5–18 mse-points. Within no-space, capitalised tokens are
much worse than lowercase. The interpretation is straightforward:
**leading-space tokens correspond to actual word boundaries in the source
text** and tend to carry well-formed semantic content; no-space pieces are
either continuations of multi-token words *or* the very first token of a
sentence/title (the most common reason a capital appears with no leading
space). Sentence-starts have less left-context to disambiguate against.

## 3. Worst-case examples — they all look alike

Of the 34 catastrophic failures (non-sink, cos < 0.7), **28 are in
`cap_no_space`** (the rest are split between `subword_continuation`,
`single_char_alpha`, and `punct_only`; one of each).

The 28 cap_no_space failures share *every* common feature:

| feature | value |
|---|---|
| token | `How`, `What`, `Android`, `Any`, `VPN`, `Changing`, `Should`, `Self`, `Show`, `Up`, `Why`, `Implement`, `Distinct`, `Th`, `Node`, `Time`, `Unable`, `Ps`, `Asp`, `Using`, `Device` |
| `vec_norm` | **always ≈ 36,000** (range 36,023–36,505) — the absolute lowest decile of the dataset (median is 73k) |
| `text_preview` | always begins with the same token, indicating the dataset entry is a Stack-Exchange-style title-then-body |
| AV `explanation` | almost always one of: "Technical article structure: a blog post reviewing a specific research paper…", or contains "Fibonacci sequence" / "Kalman filter" |

The position field (range 34–435) is *not* predictive — these aren't all
at position 0. They're sampled from various positions within long
documents; what unites them is the token identity and the very-low
activation norm at that position.

A representative example:

```
cos=0.143  pos=391  ‖v‖=36411  tok='How'
context (first 90 chars): "How to run .sql file using java without manually parsing the document\nHow to run .sql file…"
AV explanation: "Technical article structure: a blog post reviewing a specific research paper, with a consistent pattern of explaining the problem and solution. The phrase \"The…"
```

The activation should encode "I am the second occurrence of the word
'How' in a Stack Overflow title that's been pasted twice", but instead the
AV produces a generic templated description and the AR cannot recover the
direction.

## 4. The two AV fallback templates

Section 10 of the auto-generated report finds explanations that appear
verbatim in 5+ samples. The top two are:

| n verbatim copies | mean cos when this template appears | first 200 chars |
|---|---|---|
| **9** | **1.000** | `Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a collage or corrupted document, with fragmented literary/cultural references. The phrase "it'" signals` |
| **7** | **0.237** | `Technical article structure: a blog post reviewing a specific research paper, with a consistent pattern of explaining the problem and solution. The phrase "The results of the study are:" signals a st` |

These are **two different failure regimes** that the AV has memorised:

- **High-norm sink** (cos = 1.000). The AV recognises high-norm activations
  as "this isn't real content" and emits the same canned literary-collage
  description. The AR has memorised this template and inverts it
  perfectly. The reconstruction is "correct" only because both halves of
  the autoencoder have agreed on a stable degenerate code; downstream this
  channel carries zero information.

- **Low-information content** (cos = 0.237). For low-norm activations
  carrying weak signal, the AV falls back to a "research paper review"
  template. The AR has *also* memorised this template — but the
  vector it predicts for it is completely uncorrelated with the true
  activations that triggered it. Mean cos = 0.237 means these
  reconstructions are essentially random with respect to the originals.

Less common but similar: explanations containing "Fibonacci sequence"
(22 samples, cos = 0.610), "Kalman filter" (7 samples, cos = 0.613). These
look like additional fallback templates the AV has memorised for low-info
activations, distinguished by minor surface variation.

The combined volume of the two main fallbacks (15 + 9 = 24 samples) plus
the `Fibonacci`/`Kalman` patterns (29 samples) accounts for ~50 of the
~60 worst non-sink samples — i.e., the failure tail is dominated by
template emission, not by random per-sample noise.

## 5. What does *not* predict failure

Several plausible axes turned out to be flat:

- **Position in sequence.** All 10 deciles between position 10 and 511 sit
  at cos_mean ∈ [0.991, 0.994]. Position-0–9 was excluded by the extract
  script (per the README warning), and past that the activation has
  enough context for the model regardless of where in the document we
  sample.
- **Sequence length.** Identical story — 0.991 → 0.993 across deciles.
  (11k samples are at seq_len = 512 because of right-truncation, but they
  reconstruct just as well as shorter sequences.)
- **Token length.** Lengths 1–20 chars all sit in [0.992, 0.995]. The
  category effect (cap_no_space = 0.978) dominates whatever weak length
  signal exists.
- **AV explanation length.** A counter-intuitive non-result: explanations
  with 100+ words *do* reconstruct slightly better than 51–70 word ones
  (0.994 vs 0.988), but the effect is small and probably mediated by the
  same content-quality signal that drove the AV to write more in the
  first place.

## 6. Recommendations

For interpreting NLA outputs in downstream work:

1. **Skip the sink cluster.** The 11 samples (0.06 %) at vec_norm > 120k
   round-trip at cos = 1.000 but carry no information; their AV text is a
   literal memorised string.
2. **Treat AV outputs containing the strings *"Technical article
   structure: a blog post reviewing a specific research paper"* or
   *"Fibonacci sequence"* / *"Kalman filter"* as low-confidence.** They
   are AV's "I don't know" responses and the corresponding reconstructions
   are not meaningful (mean cos ≈ 0.24–0.61). A simple substring check is
   sufficient to filter these out.
3. **Capitalised, no-leading-space tokens at low L2-norm need extra
   skepticism.** The combination "cap_no_space + vec_norm < 40k" identifies
   essentially all of the 28 catastrophic-failure samples.
4. **Position and sequence length can be ignored** as quality predictors
   past position 10. The flatness here is a positive signal — the model's
   residual stream is self-organising past the warm-up window.

## 6a. Filter validation

Empirically testing the recommendations from §6 against the 32 cos < 0.5
samples in the dataset:

| filter                                                          | flagged | % of all | cos < 0.5 caught | mean cos of flagged |
|-----------------------------------------------------------------|---------|----------|------------------|---------------------|
| F1: explanation contains "research paper" / "Fibonacci" / "Kalman" | 37 | 0.18 % | 19/32 (59 %) | 0.46 |
| F2: vec_norm < 40 000 **and** token starts with capital, no leading space | 41 | 0.21 % | 27/32 (84 %) | 0.42 |
| F1 ∪ F2                                                          | 61 | 0.30 % | **30/32 (94 %)** | 0.64 |

The unflagged remainder has mean cos = **0.994** (vs. mean over the whole
dataset of 0.993). So a 0.30 %-of-data pre-filter eliminates almost every
catastrophic failure and lifts the mean cos by 0.001. The two miss cases
are unusual but rare enough to ignore at this scale.

This is the practical takeaway: **a substring match on the AV explanation
plus a (norm, surface form) check on the input is enough to reject the
unreliable region** — no need for downstream confidence calibration.

## 7. Open questions for a follow-up

- Is the "vec_norm ≈ 36k" cluster a Gemma-3 specific phenomenon (some kind
  of attention-sink lite, or LayerNorm interaction)? The norm is far too
  consistent across distinct documents to be coincidence.
- Does the AV's failure-template choice ("research paper" vs "Fibonacci"
  vs "Kalman") correlate with anything systematic about the input
  activation, or is it random sampling among learned fallbacks?
- For the cap_no_space failures: the underlying texts are all Stack
  Overflow-style. Would the same tokens at the start of e.g. a Wikipedia
  article fail the same way, or does the failure depend on the specific
  context distribution?

A 100k sweep would not change any of these qualitative findings — the
distribution is already very tight at 20k — but it would give us
~5x finer granularity for the per-token tables, which might surface
additional template patterns we haven't named yet.
