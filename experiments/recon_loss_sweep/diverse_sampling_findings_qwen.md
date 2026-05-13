# Qwen2.5-7B rerun with diverse sampling — full cross-model picture

Repeated the 20k sweep with `--sampling diverse_shards --seed 0` for Qwen,
using **the exact same 31 shards** as the new Gemma run. The activations
parquet's sidecar metadata confirms the shard list matches Gemma's
`activations_gemma12_diverse_shards_seed0_20000.parquet.meta.json` —
true apples-to-apples on identical source documents.

## TL;DR

The diverse-sampling rerun reveals an asymmetry between the two models:

- **Gemma got slightly worse** with diverse data (mean NMSE +17 %,
  catastrophic-failure count 2.4×). The biased SE-only sample had under-
  exposed Gemma's main failure mode.
- **Qwen got *better*** with diverse data (median NMSE −12 %,
  mean −8 %). The biased SE-only sample had over-exposed Qwen's main
  failure mode.

Same data, opposite responses. Each model was being mis-characterised in
opposite directions by the biased sampler.

## Four-way headline

| run                        | n     | mean   | median | p10    | p90    | p99    | cos<0.5 |
|----------------------------|-------|--------|--------|--------|--------|--------|---------|
| Gemma OLD (biased SE)      | 20000 | 0.0143 | 0.0086 | 0.0045 | 0.0245 | 0.0466 |  32     |
| **Gemma NEW (diverse)**    | 20000 | **0.0167** | 0.0083 | 0.0044 | 0.0229 | **0.0589** | **76** |
| Qwen  OLD (biased SE)      | 20000 | 0.2486 | 0.2293 | 0.1433 | 0.3793 | 0.5520 |   4     |
| **Qwen  NEW (diverse)**    | 20000 | **0.2280** | **0.2025** | **0.1212** | 0.3687 | 0.5794 |   9     |

Sampling effect per model:

| metric            | Gemma OLD → NEW         | Qwen OLD → NEW          |
|-------------------|-------------------------|-------------------------|
| mean NMSE         | 0.0143 → 0.0167 (+17 %) | 0.2486 → 0.2280 (−8 %)  |
| **median NMSE**   | 0.0086 → 0.0083 (≈ 0)   | **0.2293 → 0.2025 (−12 %)** |
| p99 NMSE          | 0.0466 → 0.0589 (+26 %) | 0.5520 → 0.5794 (+5 %)  |
| cos<0.5 count     | **32 → 76 (+138 %)**    | 4 → 9 (+125 %)          |

Both models have more catastrophic failures with diverse data, but
**only Gemma's median is unchanged while its mean inflates** (heavy tail
expansion). For Qwen, the *typical* sample lands at a much lower NMSE
under diverse sampling.

## Why Qwen improves under diverse sampling

The biased SE sample was 100 % programming Q&A — code snippets, error
messages, identifiers, API names, framework jargon. That content has:

- many short structural tokens (digits in error codes, single letters in
  identifiers, code punctuation)
- few long content words
- repeated programming-language switching that confuses AV's genre
  detection

Diverse data has prose-heavy sources (books, news, articles, recipes,
case law, parliamentary debate) where long content words dominate. Qwen's
AV handles those much better — and **every single token category got
better** with diverse sampling:

| category               | mean NMSE old → new | Δ        |
|------------------------|---------------------|----------|
| `digit_only`           | 0.3224 → 0.2896     | **−0.033** |
| `subword_continuation` | 0.2792 → 0.2471     | **−0.032** |
| `cap_no_space`         | 0.2571 → 0.2390     | −0.018   |
| `word_lower_with_space`| 0.2089 → 0.1911     | −0.018   |
| `cap_with_space`       | 0.2170 → 0.2017     | −0.015   |
| `single_char_alpha`    | 0.2446 → 0.2284     | −0.016   |
| `code_syntax`          | 0.3035 → 0.2906     | −0.013   |
| `punct_only`           | 0.2912 → 0.2809     | −0.010   |
| `whitespace`           | 0.3181 → 0.3155     | ≈ 0      |

The biggest improvements are the categories most over-represented in the
SE-biased data (digits, subword continuations from broken-up
identifiers). Whitespace is essentially unchanged — it's source-agnostic
hard for Qwen.

## Why Gemma worsens under diverse sampling

The flip side: Gemma's catastrophic failure mechanism is **document-initial
content tokens at vec_norm ≈ 36 k**, where AV emits the memorised
"research paper / Fibonacci" template. That trigger is rare in pure SE
content (which is conversational Q&A rather than title-then-body) but
common across many other sources:

- Global Voices articles (`Indonesia:`, `Nigeria:`, …)
- Recipes (`Hamburg And Rice`)
- Wikipedia categories (`Category:…`)
- Microsoft KB articles (`Microsoft KB Archive/…`)
- Case captions (`Henry Killick … Appellant, v. …`)
- YAML front-matter, shell scripts

So Gemma sees a 2.4× rise in catastrophic count, while its median is
unchanged — the median sample still reconstructs perfectly, but the
failure tail is much fatter when the data isn't all conversational SE.

## What stays the same for Qwen

The qualitative *worst-cases* picture is unchanged. The new worst-20:

```
NMSE=1.237  tok='apter'    Sausage recipe                    (broken word piece)
NMSE=1.202  tok='-'        Scientific abstract               (punctuation)
NMSE=1.165  tok='}}'       Template syntax                   (code punct)
NMSE=1.147  tok='2'        Medical paper citation            (digit)
NMSE=1.105  tok=' '        Markdown background-image style   (whitespace)
NMSE=1.058  tok='@'        Protein-kinase paper              (punct)
NMSE=1.042  tok='.,'       NLP-task definition text          (punct)
NMSE=1.037  tok='    '     PHP code namespace                (whitespace)
NMSE=1.012  tok='\n\n'     SQuAD-style question prompt       (whitespace)
NMSE=0.990  tok=' cannot'  Multi-sentence reasoning task     (mid-word)
```

Same shape as the old worst list: short / structural tokens (digits,
punctuation, whitespace, sub-word fragments) deep in long contexts. AV
produces fluent topic-confabulation. Source variety doesn't change the
mechanism — only the contexts in which it appears.

## Cross-model takeaway

The two diverse runs give the truthful cross-model comparison the biased
runs couldn't:

| metric            | Gemma diverse | Qwen diverse | Qwen/Gemma ratio |
|-------------------|---------------|--------------|------------------|
| mean NMSE         | 0.0167        | 0.2280       | 13.6×            |
| median NMSE       | 0.0083        | 0.2025       | 24.4×            |
| p10 NMSE          | 0.0044        | 0.1212       | 27.5×            |
| catastrophic rate | 0.38 %        | 0.05 %       | 0.13×            |

Gemma is still ~15× better on the typical sample, but **Qwen is 8×
*better* on catastrophic-failure rate** (0.05 % cos<0.5 vs Gemma's 0.38 %).
The Gemma autoencoder is much more accurate on average but has a
visible failure tail driven by the AV's memorised "research paper"
fallback template. The Qwen autoencoder is uniformly less accurate but
distributes its error smoothly with no fat tail.

This is the same picture I described under the cos-based view, but the
NMSE numbers make the trade-off explicit: **Gemma if you want mean-case
accuracy and can filter the failure cluster; Qwen if you can tolerate
mediocre typical-case performance in exchange for absence of
catastrophic outliers.**

## Reproducibility

Same sampler as the Gemma run. The shard list in the activations
parquet's `.meta.json` is identical between the two diverse runs (only
the base model and layer differ), so any future analysis can be replayed
exactly with:

```
python recon_loss_sweep.py extract \
  --base-model {google/gemma-3-12b-it | Qwen/Qwen2.5-7B-Instruct} \
  --layer {32 | 20} --n 20000 --max-len 512 \
  --batch-size {8 | 16} \
  --sampling diverse_shards --seed 0 \
  --out activations.parquet
```

## Files

- `data/activations_qwen7_diverse_shards_seed0_20000.parquet`
- `data/activations_qwen7_diverse_shards_seed0_20000.parquet.meta.json`
- `data/results_qwen7_diverse_shards_seed0_20000.parquet`
- `plots/top100_best_qwen_diverse.csv`, `top100_worst_qwen_diverse.csv`
- `plots/nmse_vs_token_length_diverse.png` — replaces the old biased plot
- `plots/nmse_heatmap_length_position_diverse.png` — replaces the old heatmap
