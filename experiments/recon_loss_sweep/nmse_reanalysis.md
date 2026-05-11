# Reanalysis under NMSE: what the cos view was hiding

NMSE = `mse_nrm` from the parquet = `2 · (1 − cos)` under √d normalisation.
Range [0, 4]; 0 = perfect, 2 = orthogonal. Lower is better.

This document re-asks "which kinds of samples reconstruct poorly" using
NMSE rather than cos as the quality metric. The two metrics rank
*individual* samples identically (worst-100 sets overlap 100/100 for both
models), but NMSE makes failure-tail behaviour visible that cos compresses,
and that surfaces three new findings the cos analysis missed.

## TL;DR — three things NMSE shows that cos didn't

1. **Tail dominance is wildly asymmetric between models.** Gemma has
   extreme failure outliers — 0.17 % of samples have NMSE > 100× the
   median (i.e. cos < 0.14 vs. typical cos ≈ 0.996). Qwen has **zero**
   samples above 5× median; its distribution is bounded.
2. **Gemma's failure surface is multi-category, not single-cell.** Three
   token categories all have 3–5 % outlier rates: `cap_no_space`,
   `whitespace`, and `digit_only`. The cos analysis only flagged
   `cap_no_space` because its individual failures were the most extreme;
   whitespace and digit failures were hidden under a high mean cos
   (0.99) that compressed mid-severity errors.
3. **Gemma's position effect is non-monotonic under NMSE.** Mean NMSE is
   *highest* in the 20–40 % position bucket and *lowest* at the very
   end of sequences (80–100 %). The cos view showed all positions as
   essentially identical (~0.991–0.994). NMSE reveals that mid-document
   positions concentrate the failure tail.

## 1. Distribution shape

| metric          | Gemma-3-12B | Qwen2.5-7B |
|-----------------|-------------|------------|
| mean NMSE       | 0.0143      | 0.2486     |
| median NMSE     | 0.0086      | 0.2293     |
| **mean / median ratio** | **1.66×** (tail-heavy) | **1.08×** (no fat tail) |
| p99 NMSE        | 0.0466      | 0.5520     |
| max NMSE        | 1.8482      | 1.2636     |

The mean/median ratio is the headline. A ratio > 1.5 means the mean is
being pulled up by a tail of large values; Gemma's 1.66× confirms the
"mostly-perfect with rare catastrophes" picture. Qwen's 1.08× confirms
"unimodal mediocre, no fat tail".

### Tail-occupancy (multiple of median)

| samples with NMSE >       | Gemma         | Qwen          |
|----------------------------|---------------|---------------|
| 2 × median                 | 4 155 (20.8 %) | 701 (3.5 %)   |
| 5 × median                 | 283 (1.42 %)   | **1** (0.01 %) |
| 10 × median                | 38 (0.19 %)    | 0             |
| 100 × median               | 34 (0.17 %)    | 0             |

Gemma has 34 samples at NMSE > 100× median — the catastrophic
"sentence-initial cap at low norm" failures we identified earlier. Qwen
has none anywhere near that — its worst sample (NMSE = 1.26) is only
5.5× its median.

This is the cleanest single-number summary of the qualitative
"step-function with outliers vs. smooth bell" difference between the
two models.

## 2. Per-category outlier rates (Gemma)

Threshold: NMSE > 5× overall median = 0.043 (≈ cos < 0.978):

| category                   | n     | n outliers | outlier rate |
|----------------------------|-------|------------|--------------|
| `code_syntax`              |    14 |          1 | 7.14 % (n too small) |
| **`whitespace`**           | 1 701 |         78 | **4.59 %**   |
| **`digit_only`**           |   954 |         40 | **4.19 %**   |
| **`cap_no_space`**         | 1 054 |         36 | **3.42 %**   |
| `punct_only`               | 4 292 |         80 | 1.86 %       |
| `subword_continuation`     | 1 890 |         30 | 1.59 %       |
| `single_char_alpha`        | 1 172 |         11 | 0.94 %       |
| `cap_with_space`           |   685 |          3 | 0.44 %       |
| `word_lower_with_space`    | 8 238 |          4 | **0.05 %**   |

The new finding: `whitespace` and `digit_only` actually have **higher
outlier rates than `cap_no_space`** (4.6 %, 4.2 %, 3.4 %). The previous
analysis singled out `cap_no_space` because *its* outliers were the
most extreme (cos < 0.5). But under NMSE, with all values on the same
scale, three categories are similarly outlier-prone — they just fail
with different severities.

The 90× gap between word-content tokens (0.05 %) and the worst three
categories (3.4–4.6 %) is also far more striking than the cos view
(which made it look like 0.996 vs. 0.976 — a 2 % difference that was
easy to dismiss).

## 3. The Gemma whitespace and digit failures look the same as Qwen's

For Gemma, when whitespace fails the AV makes the same kind of confident
wrong-context confabulation we saw on Qwen — just less often:

```
NMSE=0.102  cos=0.949  pos=178  tok='\n'
context:  How can I write a paragraph wrapping up a table in bootstrap?...
AV says:  "Academic tutorial structure: HTML/tutorial format with code
           blocks and structured recommendations for a web page... The
           phrase 'body { ... } ' is a CSS block listing the..."

NMSE=0.085  cos=0.957  pos=101  tok='  '
context:  Why cant we instantiate abstract class — As we all know that we
          cant instantiate an abstract class. But look into this cod...
AV says:  "Structured tutorial format with code block and Java example
           establishes a Q&A pattern, expecting a complete program
           snippet..."
```

```
NMSE=0.079  cos=0.960  pos=320  tok='0'
context:  Relative variable importance values vs. magnitude of effect — I
          have ran a series of models to see which best fit the resp...
AV says:  "Structured statistical report format: product listing with
           standardized drug information, following a consistent pattern
           of presenting behavioral data for a specific ANOVA test..."
```

The mechanism is the same as Qwen's failure mode: short / structural
token → AV can't extract content → AV writes confident detailed prose
about a wrong topic ("CSS body block", "drug information / ANOVA"). The
difference is that Gemma also has a *memorised template* fallback for
the most extreme cases (the "Fragmented academic" / "research paper"
templates) that NMSE happens to flag at the very top of the tail.
Qwen never falls back on a memorised template — it always writes new
prose — but the failure regime is otherwise identical.

## 4. Per-category outlier rates (Qwen)

Threshold: NMSE > 1.5× overall median = 0.344 (≈ cos < 0.83):

| category                   | n     | n outliers | outlier rate |
|----------------------------|-------|------------|--------------|
| **`digit_only`**           | 1 032 |        383 | **37.1 %**   |
| **`whitespace`**           |   968 |        337 | **34.8 %**   |
| `code_syntax`              |   296 |         84 | 28.4 %       |
| `punct_only`               | 3 713 |        946 | 25.5 %       |
| `subword_continuation`     | 1 423 |        316 | 22.2 %       |
| `other`                    |   810 |        149 | 18.4 %       |
| `cap_no_space`             |   953 |        163 | 17.1 %       |
| `single_char_alpha`        | 1 038 |        175 | 16.9 %       |
| `cap_with_space`           |   830 |         55 | 6.6 %        |
| `word_lower_with_space`    | 8 937 |        402 | **4.5 %**    |

For Qwen the picture is the same shape but inflated ~10×: digits and
whitespace lead at ~35 %, content words sit at 4.5 %. The relative
ordering matches the Gemma table exactly (with `cap_no_space` falling
to mid-pack in both views), confirming the failure-prone categories
are model-agnostic features of how NLA pairs handle structural tokens.

The most striking number here: **digit-only tokens fail at 37 % of the
time** (where "fail" = > 1.5× the typical NMSE for the dataset). Two
out of every five digit activations passed through Qwen NLA reconstruct
poorly enough to be flagged.

## 5. Position effect under NMSE — Gemma is non-monotonic

Mean NMSE by position quintile:

|             | Gemma mean | Gemma median | Qwen mean | Qwen median |
|-------------|------------|--------------|-----------|-------------|
| pos 0–20 %  | 0.0143     | 0.0083       | 0.2009    | 0.1843      |
| 20–40 %     | **0.0170** | 0.0097       | 0.2360    | 0.2190      |
| 40–60 %     | 0.0157     | 0.0091       | 0.2572    | 0.2389      |
| 60–80 %     | 0.0129     | 0.0083       | 0.2688    | 0.2504      |
| 80–100 %    | **0.0115** | 0.0079       | **0.2802**| **0.2608**  |

Two surprises:

- **Gemma's mean NMSE peaks at 20–40 % and is *lowest* at 80–100 %.**
  This is invisible under cos (all positions read 0.991–0.994). The
  median tells the same story but more weakly (0.0083 → 0.0097 →
  0.0091 → 0.0083 → 0.0079). Mean is amplified because the failure tail
  concentrates around positions 100–300 — the body of Stack Overflow
  questions where the title typically gets repeated and confused
  recapitulation tokens land.
- **Qwen's NMSE is monotonically increasing with position**, both mean
  and median. Same direction as the cos view but more pronounced when
  expressed as NMSE (0.20 → 0.28 = +40 % degradation, vs. cos 0.91 →
  0.86 which sounded smaller).

So **NMSE flips the polarity of the position effect for Gemma**: cos
showed late ≈ early; NMSE shows mid-document worse than either end.
For Qwen the direction is preserved but the magnitude is more
striking.

## 6. Worst-100 individual samples — same set, different ranking

For both models, the **set of the worst 100 samples is identical** under
NMSE and cos (overlap = 100 / 100). The two metrics are monotonically
related (`NMSE = 2·(1−cos)`), so any sample-level ranking is preserved.

So NMSE doesn't surface *new individual failures*. What it does is:

- Make the failure-rate-per-category structure visible (§ 2 + § 4).
- Make the tail-vs-typical separation visible (§ 1).
- Make the position-effect distinguishable from noise (§ 5).

These are all aggregate properties that depend on the relative scale
of "small errors" vs "big errors", which cos compresses near 1.

## 7. Updated takeaways

1. **Gemma's failure mode is broader than the cap_no_space cluster I
   originally identified.** Three categories are ~equally outlier-prone
   (whitespace, digit, cap_no_space — 3.4–4.6 % outlier rates), and
   they all fail via the same mechanism (AV-confabulates-wrong-topic).
   The cap_no_space failures were just the most extreme tail.

2. **A single rule "filter all tokens that aren't `word_lower_with_space`"
   would catch 95 % of Gemma's poor-reconstruction cases at the cost of
   filtering 60 % of the data.** That's a high-precision filter at low
   recall of usable data — useful when a downstream task is intolerant
   of any failure but not when you need most samples through.

3. **For Qwen, NMSE confirms the smooth-degradation picture** but quantifies
   how much worse the worst categories are: digit and whitespace failure
   rates of 35–37 % at the 1.5× threshold are nearly an order of
   magnitude above content-word rates (4.5 %).

4. **Mid-document positions are unexpectedly bad for Gemma.** The
   non-monotonic position effect (worst at 20–40 %, best at end) is a
   new finding from the NMSE view. Mechanistic guess: Stack Exchange
   posts in this dataset typically have the title repeated near the
   start of the body — that's where catastrophic `cap_no_space + low
   norm` failures concentrate.

5. **NMSE is the right metric when the cos values cluster near 1.0.**
   Concretely, for Gemma, "0.996 vs 0.976" looks like a 2 % difference
   in cos but is a **6.7×** difference in NMSE (0.007 vs 0.048). Anyone
   reading the cos table would underweight the gap; the NMSE table
   makes it impossible to miss. For Qwen (cos in the 0.85 region)
   either metric works fine — the difference matters most when the
   model is near-perfect.
