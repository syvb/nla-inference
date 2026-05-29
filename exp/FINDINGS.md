# NLA Verbalization & Reconstruction — Experiment Summary

*An investigation into where the Natural Language Autoencoder's reconstruction
power comes from, and whether it can be improved at inference time.*

---

## 1. Background: what is an NLA?

A **Natural Language Autoencoder (NLA)** turns an internal activation vector of
a language model into English and back. It has two halves:

- **AV — Activation Verbalizer** (`...-av` checkpoint): takes a hidden-state
  vector `h` (here, layer-32 residual stream of Gemma-3-12B) and emits a short
  English **explanation** of what that vector "means" — a 2-3 paragraph
  analysis of what the model is predicting at that point in the text. The
  vector is fed in by *injection*: a marker token's embedding is overwritten
  with the (rescaled) activation.
- **AR — Activation Reconstructor** (`...-ar` checkpoint): takes the
  explanation text and predicts the original activation vector `ĥ`.

The round trip `h → AV → text → AR → ĥ` is an autoencoder with a **natural
language bottleneck**. Reconstruction fidelity (how close `ĥ` is to `h`) tells
you how much of the vector's content the English explanation captured.

**Reconstruction metric.** Throughout we use `mse_nrm`: both vectors are
L2-normalized to norm √d (d=3840), then mean-squared-error per dimension. This
equals `2(1 − cos)`, i.e. it is *direction only* — magnitude is discarded.
Lower is better; 0 is perfect, ~2 is orthogonal.

**FVE (fraction of variance explained).** The paper reports quality as
`FVE = 1 − E‖h − ĥ‖² / E‖h − h̄‖²`, where `h̄` is the mean activation. FVE=0
means "no better than always predicting the mean activation", FVE=1 is perfect.
Some tables below instead report **FVE-over-empty**, which uses the AR run on
an *empty* explanation as the 0-point — a slightly weaker baseline than the
mean activation (empty-prompt mse ≈ 0.052 vs mean-activation mse ≈ 0.031 on
chat), so FVE-over-empty numbers run a bit higher than paper-style FVE.

**Datasets.**
- **chat** — WildChat conversations (`results_chat_20k.parquet`), assistant
  turns regenerated with Gemma-3 at max_length 2048. 19,881 activations.
- **PT (pretraining)** — `common-pile/comma_v0.1_training_dataset`, sampled
  across 31 diverse shards (arxiv, pubmed, caselaw, github, …).
  20,000 activations.

---

## 2. The core question

**How much of the AV's reconstruction ability is "real verbalization" vs.
something cheaper?** Two sub-questions drove the work:

1. **How much does the trained AV beat its own warm-start initialization?**
   The AV is bootstrapped (SFT) from explanations a frontier model (Sonnet 4.6)
   produces with a fixed "warm-start" prompt (`stage2_api_explain.py`), then
   improved with RL. If a frontier model with the same prompt nearly matches
   the trained AV, the AV's value is mostly "knowing the format". If not, the
   AV is doing something the frontier model can't — most plausibly *using the
   injected activation*.

2. **Is the AR operating at its information ceiling?** If we hand the AR *more*
   context at inference time, can it reconstruct better — with no retraining?

---

## 3. Thread A — Warm-start vs. trained AV

We re-created the warm-start prompt verbatim and asked **Sonnet 4.6** (via
OpenRouter) to verbalize, using the *source text prefix* as input (the AV sees
the activation; Sonnet sees the text the activation came from). Then we ran the
AR critic on Sonnet's explanations and compared to the AV's.

A key early correction: the activations parquet only stores a 50-char
`chars_before` snippet. Using that as Sonnet's context (v1) badly handicapped
it. We rebuilt the **full prefix** by re-rendering the chat template /
re-tokenizing the source doc and slicing to the activation position (v2+),
matching exactly what the extractor saw (verified: last-token identity
matches in 99.8% of rows).

**Chat warm-start progression (FVE-over-empty, n≈485):**

| variant | FVE-over-empty |
|---|---|
| empty prompt | 0.000 |
| v1 — Sonnet, 50-char prefix (buggy) | 0.272 |
| v2 — Sonnet, full prefix | 0.554 |
| v3 — + explicit final-token hint | 0.560 |
| v4 — + 4 few-shot examples | 0.578 |
| **AV (trained)** | **0.815** |

**Findings.**
- With proper full context, Sonnet recovers ~2/3 of the AV's signal
  (0.55 / 0.81). The remaining third is what the activation injection (plus RL)
  buys.
- **The final-token hint barely helped** (+0.006). The AR isn't keyed on the
  literal token; it reads the *analysis*.
- **Few-shot demonstrations of the AV's style added only ~0.024 FVE**, and
  going from 4→10 shots added almost nothing more. So the AV's edge is *not*
  mostly "knows a better verbalization structure" — it's the activation.
- Prompt engineering on Sonnet (system-prompt restructure, "quote the source
  text verbatim" instruction — the hand-written `prompt.txt`) pushed PT
  warm-start to FVE-over-empty 0.661, still ~15 pp short of the AV's 0.810.

The gap between a strong frontier model with full text and the trained AV is
real and stubborn — consistent with the activation carrying information not
recoverable from the text alone.

---

## 4. Thread B — Can the AR do better with more context? (the `av_delim` result)

We then asked the flipped question: give the **AR** both the explanation *and*
the source text. Several formats were tried (PT, n≈351):

| format fed to AR (inside `<text>…</text>`) | FVE-over-empty | beats AV |
|---|---|---|
| **AV (baseline — explanation only)** | **0.807** | — |
| input_only (raw source text, no analysis) | 0.364 | 0.9% |
| v7_swap (Sonnet analysis, then source) | 0.629 | 13% |
| v7delim (source, then Sonnet analysis) | 0.663 | 29% |
| av_delim_swap (AV analysis, then source) | 0.790 | 48% |
| **av_delim (source, then AV analysis)** | **0.834** | **94.3%** |

The winning format:

```
Original text:
<full source prefix>

Analysis:
<AV explanation>
```

**Findings.**
- **`av_delim` beats the AV baseline by ~14% in mse** (0.834 vs 0.807 FVE),
  winning on **94.3%** of rows — *with no retraining*, just a richer prompt at
  inference time. The AR was **not** at its information ceiling.
- **Order matters.** The analysis must come **last**, immediately before the
  AR's `</text> <summary>` extraction point. Putting raw text last
  (`*_swap`) consistently hurts — the AR partially shifts into
  "summarize this text" mode instead of "reconstruct from this analysis".
- **Raw text alone is weak** (FVE 0.364): the AR is keyed on AV-style
  structured analysis, not a general text encoder. The source text *supplements*
  the analysis; it can't replace it.

### Scaling to the full chat set

Repeated on all 19,832 parseable chat rows on a single H100 (~52 min):

| | mse_nrm (mean) | median |
|---|---|---|
| AV baseline | 0.0096 | 0.0074 |
| **av_delim** | **0.0082** | **0.0062** |

- **av_delim beats the AV baseline on 91.0%** of 19,832 rows — the small-sample
  PT result (94.3%) holds at 40× the data.
- Paper-formula aggregate FVE (mean-activation baseline): AV 0.677 →
  av_delim 0.723.

---

## 5. Thread C — The symmetric test: give the *AV* the source text

If handing the AR the source text helps, does handing the **AV** the source
text (in its prompt, alongside the injected activation) help its verbalization?
We modified the AV prompt to include the source prefix before the `<concept>`
injection marker and regenerated explanations (PT, n=352).

| | FVE-over-empty |
|---|---|
| AV canonical (no source in prompt) | 0.809 |
| AV with source text in prompt | 0.798 |

**Finding: it slightly *hurts*** (better on only 48% of rows — a wash, trending
negative). The symmetry does **not** hold. Interpretation:

- The AR *encodes* text → vector and has slack to absorb extra delimited
  context.
- The AV *generates* text from a fixed prompt distribution; adding content
  shifts it off the trajectory it was trained on. The AV is already at its
  ceiling **for its current prompt format** — to actually exploit the source
  text in verbalization you'd need to retrain the AV on source-augmented
  prompts.

---

## 5b. Minor — the AV as a plain text explainer

A quick variant of Thread A: instead of asking *Sonnet* to verbalize from
text, use the **AV itself**, fed the source text in place of the injected
activation (minimal prompt rewrite: "activation vector" → "text snippet", put
`decoded_full` in `<concept>`, no injection). The thought was that the AV is
already tuned to write in exactly the AR's preferred format, so it might beat
Sonnet at text→explanation. It does not (chat, n=4820):

| | mse_nrm | FVE-over-empty |
|---|---|---|
| AV on activation (baseline) | 0.0116 | 0.780 |
| **AV on text (this variant)** | 0.0374 | **0.290** |
| *(ref) Sonnet 4.6 on text (chat v2)* | ~0.024 | ~0.554 |

AV-on-text is worse than *both* the AV-on-activation baseline and Sonnet on the
same text, and beats the activation baseline on just 0.5% of rows. The AV is
doubly handicapped on raw text: it's out of distribution (never saw text in the
`<concept>` slot) and it's only a 12B model, well below Sonnet as a general
explainer. Being tuned to the output format doesn't compensate. A clean
negative that reinforces §6.1: the AV's value is in *reading the activation*,
not in knowing the format.

---

## 6. Headline takeaways

1. **The activation injection is doing real work.** A frontier model (Sonnet
   4.6) with the full source text and heavy prompt engineering still lands
   ~15 pp of FVE short of the trained AV. Few-shot format demonstrations close
   almost none of the gap. The AV's advantage is the activation, not the
   format.
2. **The AR is not at its information ceiling.** Feeding it the source text
   *and* the AV explanation, in the right order, improves reconstruction on
   ~91-94% of samples and ~14% in mean error — for free, at inference time.
3. **Ordering / prompt-mode effects are large.** The same content helps or
   hurts depending on whether the AV-style analysis sits at the AR's extraction
   point. This is a property of how the AR was trained (analysis as the sole
   `<text>` content), not a generic context-length effect.
4. **The encode/decode asymmetry is informative.** Extra context helps the
   text→vector direction (AR) but not the vector→text direction (AV), under the
   current fixed prompts.

---

## 7. Methods & infrastructure notes

- **Models.** AV/AR = `kitft/nla-gemma3-12b-L32-{av,ar}` (~24 GB bf16 each).
  AR critic = first 33 layers + Identity final-LN + a `value_head` Linear(d,d),
  extraction at the last token (`</text> <summary>` suffix).
- **GPUs.** Started on Lambda (A100/H100/8×V100), moved to RunPod
  (`rest.runpod.io/v1`, H100/H200). Each job: launch pod → install deps →
  `snapshot_download` checkpoint → run → copy results back → **delete pod**.
- **Two bugs worth remembering:**
  - *Double embedding scale.* `model.get_input_embeddings()` for Gemma-3 is a
    `ScaledWordEmbedding` whose `forward()` already multiplies by √d. Multiplying
    again (as standalone injection code must, since it loads the bare weight)
    produces √d²-too-large embeddings → pure gibberish output. Fixed by using
    the layer's built-in scaling.
  - *Right-truncation of long augmented prompts.* `av_delim` prompts can exceed
    the AR's window; `truncation_side="left"` preserves the suffix tokens the
    AR extracts at.
- **OpenRouter prompt caching.** System/few-shot prefixes are cached with
  `cache_control: ephemeral`; **two sequential warm calls** are needed before
  firing concurrent requests (Anthropic's first cache write isn't queryable
  until just after the response returns). Achieved ~85-92% cache hit rates.

## 8. File index (`exp/`)

- `warmstart_build_prompts_v2.py` — reconstruct full chat prefixes from
  `regen_wildchat_20k.jsonl` + activations.
- `warmstart_build_pt_prompts.py` — PT equivalent (matches docs to the 31
  common-pile shards by 200-char preview).
- `warmstart_call_openrouter*.py` — Sonnet warm-start callers (multi-turn
  few-shot, system-prompt, hand-written-prompt-from-file variants).
- `warmstart_augment_modes.py`, `build_chat_av_delim.py` — build the
  source+analysis augmented inputs for the AR.
- `warmstart_run_ar_multi.py` / `warmstart_run_ar_long.py` — AR critic eval
  (multi-variant, long-context).
- `av_with_source_generate.py` — AV generation with source text in the prompt
  (§5).
- `av_on_text_generate.py` — AV run as a plain text explainer, no injection
  (§5b).
- `plot_av_vs_avdelim_*.py`, `plot_av_on_text_hist.py` — mse and FVE histograms.
- `figures/` — generated charts. `prompt.txt` — the hand-written v7 system
  prompt.
- Results parquets live under `exp/results/warmstart/` (gitignored — large).
