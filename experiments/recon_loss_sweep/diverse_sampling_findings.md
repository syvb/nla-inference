# Gemma rerun with diverse (cross-source) sampling

Repeated the 20k Gemma-3-12B / L32 sweep using the new
`--sampling diverse_shards --seed 0` mode, which picks one random shard per
top-level source directory (31 shards total) and interleaves them with a
50k shuffle buffer. The previous run (`results_20000.parquet`) was
inadvertently biased: HF's default `streaming=True` reads shards
alphabetically, and a 10k-row buffer can't escape a single multi-GB shard.
That run was essentially 100 % Stack-Exchange-style content from the first
shard of `github_archive`.

## Shards used (reproducible from `seed=0`)

All 31 source dirs got one random shard:
```
arxiv_abstracts/arxiv_abstracts.chunk.29.jsonl.gz
arxiv_papers/arxiv_papers.chunk.49.jsonl.gz
biodiversity_heritage_library/biodiversity_heritage_library.chunk.33.jsonl.gz
caselaw_access_project/caselaw_access_project.chunk.40.jsonl.gz
cccc/cccc.chunk.07.jsonl.gz
data_provenance_initiative/data_provenance_initiative.chunk.43.jsonl.gz
doab/doab.chunk.32.jsonl.gz
foodista/foodista.chunk.32.jsonl.gz
github_archive/github_archive.chunk.45.jsonl.gz
library_of_congress/library_of_congress.chunk.13.jsonl.gz
libretexts/libretexts.chunk.17.jsonl.gz
news/news.chunk.48.jsonl.gz
oercommons/oercommons.chunk.02.jsonl.gz
peS2o/peS2o.chunk.15.jsonl.gz
pre_1929_books/pre_1929_books.chunk.05.jsonl.gz
pressbooks/pressbooks.chunk.59.jsonl.gz
project_gutenberg/project_gutenberg.chunk.45.jsonl.gz
public_domain_review/public_domain_review.chunk.50.jsonl.gz
pubmed/pubmed.chunk.55.jsonl.gz
python_enhancement_proposals/python_enhancement_proposals.chunk.06.jsonl.gz
regulations/regulations.chunk.01.jsonl.gz
stackexchange/stackexchange.chunk.42.jsonl.gz
stackv2_edu/stackv2_edu.chunk.32.jsonl.gz
stackv2_html/stackv2_html.chunk.47.jsonl.gz
ubuntu_irc/ubuntu_irc.chunk.57.jsonl.gz
uk_hansard/uk_hansard.chunk.30.jsonl.gz
usgpo/usgpo.chunk.13.jsonl.gz
uspto/uspto.chunk.37.jsonl.gz
wikimedia/wikimedia.chunk.44.jsonl.gz
wikiteam/wikiteam.chunk.39.jsonl.gz
youtube/youtube.chunk.14.jsonl.gz
```

Sidecar JSON written next to the activations parquet has this list plus seed,
n, sampling mode, base model, layer — enough to reproduce the exact sample.

## Headline comparison

| metric                  | old (biased SE-only) | new (diverse, 31 sources) | Δ        |
|-------------------------|----------------------|---------------------------|----------|
| mean NMSE               | 0.0143               | 0.0167                    | +17 %    |
| **median NMSE**         | **0.0086**           | **0.0083**                | **−4 %** |
| p10 NMSE                | 0.0045               | 0.0044                    | ≈ 0      |
| p90 NMSE                | 0.0245               | 0.0229                    | −7 %     |
| p99 NMSE                | 0.0466               | 0.0589                    | +26 %    |
| max NMSE                | 1.85                 | 1.90                      | ≈ 0      |
| cos < 0.5 count         | 32                   | **76**                    | **2.4×** |
| cos < 0.7 count         | 34                   | **78**                    | **2.3×** |
| mean vec_norm           | 72 607               | 68 242                    | −6 %     |
| samples > 100× median   | 34 (0.17 %)          | **78 (0.39 %)**           | **2.3×** |

**The median is statistically indistinguishable**, but **the catastrophic
failure rate more than doubled** when we sample fairly across sources.

## What the new failures look like

The 30 worst-NMSE samples in the new run share an extremely tight signature:

- **vec_norm ≈ 35 700 – 36 500** (the same low-norm cluster that dominated
  the old run's failures — but now triggered by many sources, not just SO).
- **AV explanation** is almost always the memorised template
  *"Technical article structure: a blog post reviewing a specific research
  paper, with a structured argument about the Fibonacci sequence …"* or its
  Kalman-filter variant.
- **Token** is whatever string starts the document body, dropped onto a
  position that's deep into the sequence after the document's title is
  repeated.

Examples from the new top-30 worst (cos < 0.36):

```
cos=0.049  tok='Indonesia'  ‖v‖=36401  pos=465
  text: "Indonesia: Obama 'snub' no reason to be disappointed · Global Voices ..."
  AV  : "Technical article structure: a blog post reviewing a specific
         research paper, with a structured argument about the Fibonacci
         sequence and the Fibonacci number..."

cos=0.053  tok=' Hamburg'  ‖v‖=36178  pos=403
  text: "Hamburg And Rice — By: Anonymous — Published: 2009..." (a recipe!)
  AV  : same Fibonacci-sequence template

cos=0.063  tok=' Nigeria'  ‖v‖=36413  pos=454
  text: "Nigeria: ReVoda Turns Voters Into Monitors · Global Voices..."
  AV  : Fibonacci-sequence-and-Gibbs-energy variant

cos=0.111  tok='Complete'  ‖v‖=36270  pos=478
  text: "Complete this political dialog on the topic of Newfoundland..."
  AV  : same template
```

**The new failure surface is the same mechanism we already identified**
(low-norm + Fibonacci template fallback) — just exposed by more types of
documents. The biased run only saw it for Stack Overflow title-repeats;
the diverse run sees it triggered by:

- Global Voices article titles (`Indonesia:`, `Nigeria:`, `Costa Rica:`,
  `Sri Lanka:`, `Bosnia:`)
- Recipes (`Hamburg And Rice`, listed `Ingredients`)
- Lab manuals (`Lab 6: Mutant Exploration`)
- Wikipedia category entries (`Category:Ukrainian-Jewish culture …`)
- Microsoft KB articles (`Microsoft KB Archive/883373`)
- Legal case captions
- Code/shell-script preambles (`#!/bin/sh`)
- YAML front-matter (`--- layout: fb-ad-layout …`)

All of them follow the same pattern: a *title-like phrase* that gets
repeated or echoed in the body of the document, and the activation lands
at the second occurrence at high position, with the residual stream
collapsing to the ~36k-norm "I'm in the body, the title is being recapped"
mode that the AV has only memorised one (Fibonacci/research-paper) verbalisation for.

## Per-category NMSE — new run

| category               | n    | mean   | median | p99   | cos<0.5 |
|------------------------|------|--------|--------|-------|---------|
| `cap_no_space`         |  825 | **0.0972** | 0.0106 | **1.77** | **50** |
| `whitespace`           | 1315 | 0.0294 | 0.0157 | 0.0841 | 9       |
| `code_syntax`          |    3 | 0.0259 | 0.0272 | 0.0382 | 0       |
| `digit_only`           | 1518 | 0.0212 | 0.0164 | 0.0699 | 2       |
| `single_char_alpha`    |  860 | 0.0191 | 0.0084 | 0.0593 | 5       |
| `punct_only`           | 3235 | 0.0184 | 0.0113 | 0.0705 | 8       |
| `subword_continuation` | 1520 | 0.0138 | 0.0096 | 0.0461 | 2       |
| `cap_with_space`       | 1564 | 0.0103 | 0.0083 | 0.0404 | 0       |
| `word_lower_with_space`| 9155 | **0.0076** | 0.0065 | 0.0244 | 0   |

Compared to the old run:

- `cap_no_space` mean NMSE doubled (0.048 → 0.097); its **p99 jumped 65×**
  (0.027 → 1.77). 50 of its samples have cos < 0.5 — 6.1 % catastrophic
  rate in this category alone (old run: 0.5 %).
- `word_lower_with_space` has **zero** cos < 0.5 samples (old run also
  had only 4; nearly clean). Content words are robust regardless of source.
- The order of categories by mean NMSE is **identical** to the old run.
  The failure mode is the same; the proportions just shifted.

## AV fallback template prevalence

| template                          | old count | new count | new mean cos when present |
|-----------------------------------|-----------|-----------|---------------------------|
| `Fragmented, incoherent academic` |    9      |     6     | 1.000 (degenerate sink)   |
| `Technical article structure: … research paper` | 15 | **36** | **0.293** |
| `Fibonacci sequence`              |   22      |  **33**   | 0.471                     |
| `Kalman filter`                   |    7      |    14     | 0.709                     |

The two "AV gives up" templates (`research paper` and `Fibonacci`) appear
**2-2.5× more often** in the diverse run, exactly tracking the doubling of
the catastrophic-failure count. Confirms: the failure mode is unchanged
mechanically; we just see it more because more source types contain the
title-repeats-in-body structure that triggers it.

## What did the old run *miss* qualitatively?

Two things:

1. **It missed how broad the trigger is.** The Gemma AV's
   "research-paper / Fibonacci" fallback isn't a Stack-Exchange-specific
   artefact — it fires for any document where a title-phrase appears
   later in the body at a position the residual stream interprets as
   "title recapitulation". That includes Global Voices articles, recipes,
   lab manuals, wiki categories, KB articles, etc.

2. **It under-counted the failure rate.** The true rate of catastrophic
   reconstruction on `common-pile`-like data is closer to ~0.4 % rather
   than the ~0.17 % I reported earlier, because the biased run was
   sampling exclusively from the source (`github_archive`) with the
   lowest density of these triggers.

## What stayed the same

- Median reconstruction quality is genuinely excellent: median NMSE 0.0083
  ≈ cos 0.996. The Gemma NLA pair works as advertised for the typical
  sample regardless of source.
- The per-category ordering of failure rates is unchanged.
- The 4 best samples in the old run (high-norm sink cluster with cos = 1.000)
  remain present (6 in the new run) — these are model artefacts, not data
  artefacts.
- All the filter rules from earlier analyses still apply: prefer
  `word_lower_with_space` tokens, filter low vec_norm + sentence-initial
  capitals, treat AV emissions of "research paper" / "Fibonacci" as
  suspect.

## Reproducibility

The new run is reproducible from:
- Script: `recon_loss_sweep.py` (in repo root)
- Command:
  ```
  python recon_loss_sweep.py extract \
    --base-model google/gemma-3-12b-it --layer 32 \
    --n 20000 --max-len 512 --batch-size 8 \
    --sampling diverse_shards --seed 0 \
    --out activations.parquet
  ```
- The sidecar JSON next to the parquet records the exact 31 shards used
  plus all hyperparameters.

The old biased mode is still available as `--sampling streaming_default`
for backward compatibility with the May-10 run.
