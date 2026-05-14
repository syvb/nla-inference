# Findings — paragraph ablation

## Question

NLA's AV emits 3 paragraphs per explanation:

1. **Structure framing** ("Technical Q&A forum format…")
2. **Context setup** ("The phrase 'X' signals…")
3. **Final-token continuation** ("Final token 'Y' opens a clause requiring…")

We replaced paragraphs 1 & 2 with *fixed constants* (chosen to look
structurally similar but unrelated to chat content) and fed the result back
through the AR. How much of the recon quality survives?

## Headline

The **final paragraph alone carries the overwhelming majority** of the
information that the AR uses to reconstruct activations. Mean NMSE goes
from `0.0111` → `0.0179` (+61 % relative) when paragraphs 1 & 2 are
replaced. 98.2 % of samples get worse, but the absolute degradation is
small — both numbers are firmly in the "high-quality reconstruction"
regime (cos ≥ 0.99).

Framed against the "no information" floor (NMSE = 2 when uncorrelated):
the final paragraph alone captures `(2 − 0.0179) / (2 − 0.0111) ≈ 99.7 %`
of the variance-reduction the full explanation provides. Paragraphs 1 & 2
together contribute ~0.3 % of the total signal.

## Overall stats (n = 19 880)

|              | mean   | median | p10    | p90    |
|--------------|--------|--------|--------|--------|
| NMSE orig    | 0.0111 | 0.0074 | 0.0041 | 0.0178 |
| NMSE modif   | 0.0179 | 0.0136 | 0.0083 | 0.0284 |
| Δ NMSE       | +0.0068| +0.0055| +0.0023| +0.0122|
| cos orig     | 0.9944 | 0.9963 |        |        |
| cos modif    | 0.9910 | 0.9932 |        |        |

- 19 525 / 19 880 (**98.2 %**) of samples reconstruct *worse* when the
  first two paragraphs are replaced.
- 355 / 19 880 (1.8 %) reconstruct *better* — concentrated in
  `role_marker`, see below.

## By role

| role          |     n | NMSE orig | NMSE modif | Δ       | cos orig | cos modif |
|---------------|------:|----------:|-----------:|--------:|---------:|----------:|
| asst_content  | 15 297| 0.0087    | 0.0157     | +0.0070 | 0.9956   | 0.9921    |
| user_content  |  4 254| 0.0123    | 0.0196     | +0.0073 | 0.9939   | 0.9902    |
| bos           |     27| 0.0006    | 0.0059     | +0.0053 | 0.9997   | 0.9970    |
| end_of_turn   |     58| 0.0320    | 0.0368     | +0.0049 | 0.9840   | 0.9816    |
| **role_marker**|   244| **0.1392**| **0.1224** | **−0.0168** | **0.9304** | **0.9388** |

`role_marker` is the only role where the ablation *helps*. These tokens
(`<start_of_turn>`, `user`/`model`, the trailing newline that closes the
role line) are the worst-reconstructed of any role in the original data
(cos 0.93 vs 0.99+ for content tokens). Their original AV explanations
likely include weak/misleading first-two-paragraph framing — replacing
them with the boilerplate ML constants slightly de-noises the input
that the AR sees. Worth investigating separately; it does not change the
content-token story.

The two main content roles (`asst_content` 77 % of data, `user_content`
21 %) behave nearly identically: Δ NMSE ≈ +0.007 each, both still landing
at cos ≥ 0.99.

## Distribution of Δ NMSE

- p10 = +0.0023, p90 = +0.0122 — the entire distribution of changes is
  basically positive and narrow.
- median Δ = +0.0055 — quite consistent across samples.

The bulk story is "everything got a small constant penalty," not "a few
samples blew up and the rest were fine." If the first two paragraphs
encoded niche information about specific samples, we'd expect a heavy
tail; instead, the effect is broad and modest.

## Interpretation

The AV/AR pipeline appears to compress the per-activation signal into
the *final paragraph* of the explanation — the sentence that explicitly
describes the final token and what follows it. Paragraphs 1–2 contribute
some additional signal but only marginally; they may act more as
**redundant context that helps the AR's language-model backbone settle
on the right interpretation** than as carriers of independent information.

For Gemma-3-12B-it specifically, this is consistent with what
[[deep_findings]] showed for the pretraining sweep: the AV's first
paragraph is often a "structure framing" template that has fairly low
entropy across the dataset, while the final paragraph is where the AV
actually pins down the activation.

## Caveats

- **Constant choice matters.** The two constants were picked to *look*
  like plausible-but-unrelated paragraphs (one ML/DS structure framing,
  one batch-norm-themed "sentence setup"). A different choice of
  constants — especially something semantically aligned with the actual
  chat content — could in principle help or hurt. A natural follow-up is
  to swap in two constants that match the typical assistant-turn topic
  (e.g., generic chat conversation framing) and check whether the +0.007
  gap shrinks.
- This ablation **does not** distinguish "the final paragraph contains
  all the information" from "the AR only attends to the final paragraph."
  The AR's last-token pooling means whatever conditioning it gets from
  earlier paragraphs has to survive a Gemma-like decoder's residual
  stream out to the last token — earlier paragraphs may carry signal
  that's simply not propagated.
- Test data: 20 k chat samples from regenerated WildChat-1M, Gemma-3-12B
  activations at layer 32. See [[recon_loss_chat]] runbook for the full
  pipeline.

## Files

- `data/ar_input.parquet` — (sample_idx, modified_explanation), 19 880
  rows, 2.4 MB.
- `data/ar_output.parquet` — (sample_idx, modified_recon), 19 880 rows,
  115 MB.
- `data/comparison.parquet` — joined back into main results with
  cos/NMSE columns for both original and modified, plus activations
  and both recons. 365 MB. Per-sample analysis material.
- `data/comparison_summary.txt` — the headline tables above as plain
  text from the compare.py run.
