# Findings — Qwen2.5-7B (L20) reconstruction-loss sweep

20,000 random activations from `common-pile/comma_v0.1_training_dataset`,
extracted at layer 20 of `Qwen/Qwen2.5-7B-Instruct`, round-tripped through the
NLA pair `kitft/nla-qwen2.5-7b-L20-{av,ar}`. Same pipeline as the Gemma-3-12B
run; results are best read alongside `findings.md` and `deep_findings.md`.

## TL;DR — Qwen behaves *very* differently from Gemma

The Qwen NLA pair reconstructs **substantially worse** than the Gemma-3-12B pair
across the entire distribution, but the *failure modes* are different in
character — not just less accurate, but qualitatively different.

| metric | Gemma-3-12B (L32) | Qwen2.5-7B (L20) | Δ |
|---|---|---|---|
| AV parse rate | 0.9999 | 0.9999 | — |
| **mean cos** | **0.993** | **0.876** | −0.117 |
| **median cos** | **0.996** | **0.885** | −0.111 |
| **p10 cos** | **0.988** | **0.810** | −0.178 |
| p90 cos | 0.998 | 0.928 | −0.070 |
| p99 cos | 0.998 | 0.949 | −0.049 |
| catastrophic (cos<0.5) | 32 (0.16 %) | 4 (0.02 %) | fewer for Qwen |
| poor (cos<0.7) | ~40 (0.20 %) | 101 (0.51 %) | more for Qwen |
| AV verbatim templates ≥ 5 copies | yes (2 dominant) | **none** | qualitatively diff |
| vec_norm range | 26k – 477k | 74 – 162 | 3-4 OOM smaller |

**Qwen is a worse autoencoder, but a more honest one.** Gemma achieves cos
≈ 0.99 by being nearly perfect on most samples and catastrophically failing
on a small clearly-identifiable subset; Qwen distributes its error more
uniformly and never reaches Gemma's ceiling. There is no "filter the bad
ones and the rest is clean" recipe that gets Qwen close to Gemma — the bulk
of Qwen samples sit at cos ≈ 0.88, not cos ≈ 0.99.

## 1. The cos distribution

```
cos quantile        Gemma     Qwen
  p10               0.988    0.810
  p25 (≈)           0.992    0.853
  p50 (median)      0.996    0.885
  p75 (≈)           0.997    0.910
  p90               0.998    0.928
  p99               0.998    0.949
```

The Qwen distribution sits ~10–18 points below Gemma at every quantile. The
spread (p90 − p10) is also wider for Qwen (0.118 vs 0.010 for Gemma) — Gemma
is essentially a step function (mostly-perfect with a few outliers), while
Qwen is a smooth distribution centered around cos = 0.88.

## 2. AV behavior — no memorised fallbacks

The most striking qualitative difference. Gemma's AV had two dominant
verbatim-repeated outputs (the "Fragmented…academic" template at 9 copies
and the "Technical article structure: research paper" template at 7).
Qwen's AV has **zero verbatim duplicates** above 5 copies. Every
explanation it produces is at least slightly different from every other.

This means the Qwen AV is *trying harder* on every sample — it doesn't
have a "I don't know" fallback to fall back on. The cost is that none of
its explanations are perfect, but the gain is that none of them collapse
to a memorised string with no information content.

The "Fibonacci sequence" / "Kalman filter" minor-fallback templates that
showed up rarely in Gemma also appear at the same low frequency in Qwen
(0.05 % / 0.03 %), and don't materially affect cos when present
(cos = 0.85 / 0.88 vs overall 0.876). These look like model-agnostic
tropes the underlying base models share rather than NLA-specific fallbacks.

## 3. Norm distribution — no sink cluster

Qwen activations are in a tight band of 74–162 (the README's predicted
"100–170 L2-norm band"). There is no high-norm tail at all — the maximum
norm in 20k samples is 162.3 — so there is no sink-token cluster to
characterise. Gemma had 11 samples ≥ 120k (and the max was 477k).

This kills the "high-norm + memorised template = perfect-but-degenerate
reconstruction" pathology that Gemma had. For Qwen the entire dataset is
essentially in-distribution.

## 4. Position effect — monotonic for Qwen, flat for Gemma

```
position decile        Gemma cos    Qwen cos
  10 - 42                0.994       0.906
  42 - 75                0.994       0.893
  75 - 111               0.994       0.886
  111 - 148              0.991       0.878
  148 - 189              0.992       0.873
  189 - 234              0.992       0.869
  234 - 290              0.991       0.867
  290 - 349              0.991       0.865
  349 - 423              0.991       0.860
  423 - 511              0.991       0.860
```

For Gemma, position past 10 was essentially noise (cos varied in [0.991,
0.994]). For Qwen, **earlier-in-sequence positions reconstruct
meaningfully better** — the first decile (positions 10–42) averages
cos = 0.906 while the last (423–511) averages 0.860, a 4.6 percentage
point monotonic drop.

The likely explanation: at low positions, the activation summarises a
short, focused chunk of text (e.g. a Stack Overflow title) and the AV
can capture that. At late positions, the activation has integrated
hundreds of tokens of body text — too much information for a 200-token
explanation to losslessly express.

## 5. Token category ranking — different from Gemma

| category                    | Gemma cos_mean | Qwen cos_mean |
|-----------------------------|----------------|---------------|
| `digit_only`                | 0.990          | **0.839**     |
| `whitespace`                | 0.990          | **0.841**     |
| `code_syntax`               | n/a            | 0.848         |
| `punct_only`                | 0.992          | 0.854         |
| `subword_continuation`      | 0.991          | 0.860         |
| `cap_no_space`              | **0.976**      | 0.871         |
| `single_char_alpha`         | 0.993          | 0.878         |
| `cap_with_space`            | 0.995          | 0.891         |
| `word_lower_with_space`     | **0.996**      | **0.896**     |

For **Gemma** there was one clear outlier (`cap_no_space` at 0.976 — sentence
starters like `How`, `What`, `Android` failing). For **Qwen** the worst cells
are the structural / non-content tokens (digits, whitespace, code syntax,
punctuation), and the best is the same as Gemma (`word_lower_with_space`).
The spread is also much wider for Qwen (0.057 vs 0.020 between worst and
best categories).

A reasonable read: Qwen's AV cares about *content* — it can verbalise
content words well, but for tokens that are syntactic or structural the
verbalisation mostly recapitulates the surrounding text and the AR can't
recover the specific direction.

## 6. Case × leading-space — same direction, smaller magnitude

| case_ws                     | Gemma cos | Qwen cos |
|-----------------------------|-----------|----------|
| `lead-space lower`          | 0.996     | 0.896    |
| `lead-space Cap`            | 0.995     | 0.893    |
| `no-space lower`            | 0.991     | 0.857    |
| `no-space Cap`              | **0.978** | 0.870    |

The lead-space-vs-no-space effect is preserved for Qwen, but the
sentence-initial-cap-with-no-space failure mode that dominated Gemma's
worst cell is **gone**. For Qwen, lowercase-no-space is *worse* than
capitalised-no-space (the opposite of Gemma) — likely because mid-BPE-piece
lowercase tokens (`ing`, `tion`, `ed`) carry less standalone meaning
than the capitalised single tokens that Qwen actually saw a lot of in
training (initialisms, brand names, code identifiers).

## 7. Worst-case examples

The cos < 0.5 tail is tiny (4 samples) and not concentrated in any one
token category:

```
cos=0.368  digit_only         tok='7'        ‖v‖=112  pos=123
cos=0.481  punct_only         tok="''"       ‖v‖=122  pos=491
cos=0.490  subword_cont.      tok='ab'       ‖v‖=124  pos=118
cos=0.500  word_lower_w_space tok=' stub'    ‖v‖=137  pos=505
```

These are isolated weird tokens at unremarkable norms. There is no shared
signature like Gemma's "norm ≈ 36k + sentence-initial cap" cluster. The
expanded cos < 0.7 tail (101 samples) is dominated by punctuation and
digit-only tokens — but the bad rates within those categories are 0.78 %
and 2.03 %, not the 2-3 % concentration we saw for Gemma's `cap_no_space`.

## 8. Single-char alpha — which letters fail

| Gemma worst | cos | Qwen worst | cos |
|---|---|---|---|
| `'C'` | 0.930 | `'h'` | 0.766 |
| `'i'` | 0.957 | `'q'` | 0.770 |
| `'z'` | 0.986 | `'R'` | 0.797 |
| `'O'` | 0.988 | `'u'` | 0.797 |

Different letters, but the *shape* is similar: single-character alphabetic
tokens are systematically harder than multi-character word tokens for both
models. For Qwen the gap is much wider — single-char averages 0.878 vs
word-with-space at 0.896 (only 0.018 spread), but the worst single chars
drop to 0.77 (0.10+ below the category mean).

## 9. AV explanation length

For Qwen, longer AV explanations correlate with better reconstruction
(cos 0.81 at 51-70 words, 0.89 at 100+). The same effect existed for
Gemma but the gap was small (0.99 → 0.99). For Qwen the gap is meaningful
(0.81 → 0.89) — when the Qwen AV emits a short explanation, that's a
strong signal it's struggling.

## 10. Practical recommendations

1. **Qwen needs higher cos thresholds for "trusted reconstruction".** A
   sample with cos = 0.95 is a typical Gemma sample but a top-1 % Qwen
   sample. Use model-relative percentiles, not absolute cos cutoffs, when
   filtering.
2. **No simple AV-text filter for Qwen.** The two-line filter that caught
   94 % of Gemma's catastrophic failures (substring match on
   "research paper" / "Fibonacci" + low-norm + cap-no-space) catches almost
   nothing for Qwen — the failure mode is too diffuse.
3. **For Qwen, prefer activations from earlier positions** (cos drops
   monotonically with position). If you have a choice, sample at
   positions 50–150 rather than 400+.
4. **For Qwen, prefer content tokens over structural tokens.** Whitespace,
   digits, code-syntax tokens reconstruct ~5-6 percentage points worse than
   `word_lower_with_space`. If the downstream task tolerates filtering, a
   simple "alphabetic with leading space" filter trims the worst cells.

## 11. Cross-model takeaway

The two models embody two different points on a quality / interpretability
trade-off:

- **Gemma-3-12B at L32** is the better autoencoder by every aggregate
  metric, but the AV achieves that score partly by emitting memorised
  templates on hard inputs. The reconstruction is "perfect" on those
  inputs but the explanation text is uninformative — a kind of
  controlled hallucination. The catastrophic failures on the
  cap-no-space + low-norm cluster represent the AV failing *visibly* to
  do this template-emission well.
- **Qwen2.5-7B at L20** never falls back on memorised templates. Every
  explanation is at least slightly novel. The cost is a uniformly lower
  ceiling (cos ≈ 0.88 vs ≈ 0.99) but the floor is also different in
  character — uniformly mediocre rather than mostly-perfect-with-rare-
  catastrophes.

Whether either is "better" depends on the downstream use. For
*automated downstream consumption* (where you just need a good vector
back), Gemma + the simple filter we developed gets you very close to
ideal at very low cost. For *interpretability* (where you want
explanation text you can actually trust to reflect the activation),
Qwen's lack of memorised fallbacks may be more useful even though its
absolute reconstruction quality is lower.
