# Qwen2.5-7B (L20): what makes reconstruction fail?

Drilling into the high-error tail of the Qwen sweep
(`results_qwen7_20000.parquet`). Reads alongside `qwen_findings.md`.

## TL;DR — failure is a clean 2-factor product

For Qwen at L20, reconstruction quality is an **additive function of two
features of the sample**:

1. **Token surface length** (worse for short).
2. **Position in the sequence** (worse for late).

The interaction surface is monotonic in both directions, with the worst
cell — short tokens deep in long sequences — sitting at cos ≈ 0.834
versus the best cell at cos ≈ 0.908.

```
mean cos by (token char-length × position quintile)

                 p0-20%  p20-40%  p40-60%  p60-80%  p80-100%
len 1             0.879    0.856    0.843    0.837     0.834
len 2             0.892    0.874    0.862    0.853     0.843
len 3             0.900    0.886    0.875    0.870     0.866
len 4-5           0.906    0.891    0.882    0.876     0.872
len 6-8           0.908    0.892    0.883    0.877     0.871
len 9+            0.908    0.894    0.881    0.883     0.874
```

The combined "short + late" cell (≤2 chars, position > 400) holds 887
samples (~4.4 % of the dataset), mean cos = 0.838 (vs whole-dataset
0.876, vs `word_with_space` 0.889).

## 1. Token kind enrichment in the failure tail

Looking at the worst-100 samples (cos < 0.70) vs the dataset background:

| token kind         | % of worst-100 | % of dataset | enrichment |
|--------------------|----------------|--------------|------------|
| `digit_only`       | **21 %**       | 5.2 %        | **4.07 ×** |
| `whitespace`       | 10 %           | 4.8 %        | 2.07 ×     |
| `single_letter`    | 9 %            | 5.2 %        | 1.73 ×     |
| `other`            | 14 %           | 7.9 %        | 1.77 ×     |
| `punct_only`       | 29 %           | 18.6 %       | 1.56 ×     |
| `short_subword`    | 5 %            | 4.8 %        | 1.05 ×     |
| `mid_word_cap`     | 1 %            | 4.8 %        | **0.21 ×** (depleted) |
| `word_with_space`  | 11 %           | 48.8 %       | **0.23 ×** (depleted) |

What unites the over-represented categories: they are all **tokens that
carry no standalone semantic content**. A digit (`'7'`) is just a piece of
a larger number; a whitespace token is between things; a single letter is
typically inside an identifier or formula; a code-syntax punctuation
token (`)`, `};`, `={`) is structural.

What unites the depleted categories: leading-space content words. These
are tokens that name a real thing on their own (` function`, ` table`,
` request`).

Notably, the `mid_word_cap` category (sentence-initial capitalised words
like `How`, `What`) — which was the dominant failure cell for Gemma — is
**five times under-represented** in Qwen's failure tail. Whatever
mechanism breaks Gemma at sentence starts does not exist for Qwen.

## 2. Hand-inspection of the worst 30 samples

Spot patterns from the actual texts:

| # | cos  | token   | what it is |
|---|------|---------|------------|
| 1 | 0.368 | `'7'`   | an HTTP status code in a Java/Jsoup error message |
| 2 | 0.481 | `"''"`  | a quoted empty string in a Stack Overflow title about screenshots |
| 3 | 0.490 | `'ab'`  | inside a Croatian word `Naslovna` (CSS Q&A) |
| 4 | 0.500 | `' stub'` | a method-stub placeholder, near end of an Android post |
| 5 | 0.502 | `'6'`   | a digit inside a CakePHP/SQL query |
| 6 | 0.530 | `'.'`   | a period in a Ruby/Project-Euler post |
| 7 | 0.551 | `'={'`  | a Django-template attribute in a Python/allauth post |
| 8 | 0.552 | `' |'`  | a pipe character in an R `read.delim` argument list |
| 9 | 0.559 | `'3'`   | a digit in a TypeScript route-params error |
|10 | 0.562 | `' the'`| in a Python tuple-list post |
|11 | 0.562 | `'O'`   | a single letter inside `WriteLine("***...")`  |
|12 | 0.577 | `' '`   | a space inside a MySQL galera-cluster post |
|13 | 0.578 | `'q'`   | inside a quadratic-CSS expression |
|14 | 0.587 | `'1'`   | inside a Stack Exchange meta post about duplicate comments |
|15 | 0.587 | `'/p'`  | inside a JSON-path query |
|16 | 0.591 | `' '`   | inside a GoJs-diagram-scroll Q&A |
|17 | 0.597 | `'in'`  | inside `'rubygems'`/`'irb.rb'` |
|18 | 0.617 | `'8'`   | digit inside a Linux Python-compile post |
|19 | 0.618 | `'}\\'`| LaTeX brace at the end of a binomial-induction proof |
|20 | 0.627 | `'begin'`| inside an astronomy/illuminance physics post |

In **every** case the token is a **structural / placeholder token** — a
digit, a punctuation mark, a single letter inside an identifier, a
multi-syllable subword, or a generic verb stem. The surrounding context
is almost always programming or math Stack Exchange.

## 3. The AV's failure mode: confident topic-confabulation

In every cos<0.7 sample I inspected, the AV produced a fluent,
detailed, technical-sounding explanation that **describes a different
topic than the actual context**. Examples:

| actual context           | AV says it's about        | cos |
|--------------------------|---------------------------|-----|
| jsoup HTTP error in Java | "Python documentation, java.util.concurrent.FutureTask" | 0.37 |
| Android activity post    | "Greek/English forum, Java for a DVD player"            | 0.50 |
| CSS+SVG Q&A              | "Emacs color scheme, JavaScript with quadrant CSS"      | 0.58 |
| C# HashTable post        | "Indian tech / C++ optimization with HashInt"            | 0.64 |
| NSData ↔ Java string     | "HTML/CSS 3D rotation, ISO/IEC encoding"                 | 0.64 |
| Pandas dataframe         | "SAS code with proc sgplot, X variable transformation"   | 0.63 |

The AV correctly identifies "this is technical / code content" but
hallucinates a **specific, plausible, wrong** topic and writes a
detailed explanation about that wrong topic. The AR then can't recover
the original direction because the explanation is essentially
unrelated to the input.

This is qualitatively different from Gemma's failure mode, where the AV
emits a verbatim memorised template (`"research paper review…"`). Qwen's
AV is more *creative* — it makes up new wrong stories every time —
which is why the worst-cases analysis found zero verbatim duplicates
above 5 copies.

## 4. AV-emitted topic correlates with cos

Categorising the AV explanation by what it claims the document is about:

| AV-claimed topic     | cos < 0.6 | cos 0.6-0.7 | cos 0.7-0.85 | cos 0.85-0.93 | cos ≥ 0.93 |
|----------------------|-----------|-------------|--------------|---------------|------------|
| "Stack Exchange / forum / Q&A" | 12 % | 18 % | 20 % | **35 %** | **41 %** |
| "code snippet / code block"    | 35 % | 25 % | 36 % | 27 %     | 26 %     |
| "documentation format"         | **24 %** | 12 % | 8 %  | 5 %     | **2 %**     |
| "mathematical / equation"      | **12 %** | 17 % | 8 %  | 4 %     | **2 %**     |
| other                          | 18 % | 29 % | 27 % | 28 %    | 28 %       |

There's a **strong sorting effect**:

- When the AV correctly says "this is a forum / Q&A post" — which is the
  truth for almost all `common-pile/comma_v0.1` examples (Stack Exchange
  dumps) — reconstruction lands in the high band.
- When the AV says "this is documentation" or "this is math reference",
  reconstruction is much worse. These claims are typically *wrong* about
  the genre, and that wrong-genre signal pulls the AR's reconstruction
  off-target.

The AV's choice between "forum talk" and "documentation/math" appears
to be a confidence proxy: the model can almost always identify a forum
post when it has enough context, but for short / late-position
activations it can't, and falls back to the more generic "this looks
like documentation" guess.

## 5. Why short + late?

A plausible mechanism, given what we see:

- The **short token** has little intrinsic identity. The activation at a
  digit `'7'` mostly carries information *about its surroundings*, not
  about itself.
- The **late position** has accumulated many hundreds of tokens of
  context. The activation is a compressed summary of all that context
  (plus the immediate token), but the local-token signal can be drowned
  out.
- Together: the AV sees an activation that's mostly a "context blob" with
  a faint "local token = digit" signal. To produce a 200-token
  explanation, it has to commit to a specific topic — and with weak
  signal, it commits to the wrong one. Confident fluent prose, wrong
  content.

The earlier AV explanation-length analysis showed shorter explanations
correlate with worse cos — consistent with this: the AV writes less when
it's confident, and writes more when it's confabulating to fill the
budget. (Wait — actually it's the opposite. Let me re-check: in §9 of
qwen_deep_analysis_20k.md, longer explanations had higher cos. So the AV
*does* produce more when it has more to say, and the failure mode is
more *aggressive over-commitment to a wrong topic* rather than rambling
to fill space.)

## 6. Practical implications

For anyone using Qwen NLA on this dataset:

1. **Discount short tokens late in the sequence.** Conservative filter:
   `n_chars(token) ≤ 2 AND position > 400` flags 887 / 20 000 samples
   (4.4 %); their mean cos is 0.838 vs. dataset 0.876.

2. **Prefer activations on `word_with_space` tokens.** The category alone
   gives mean cos = 0.896 (vs dataset 0.876, vs digit-only 0.839).

3. **AV-text "documentation format" or "math reference" claim is a
   smell.** Both are over-represented in the failure tail by ~12× and
   ~6× vs the high-cos band. A simple substring check on the AV
   explanation can flag suspicious reconstructions.

4. **Don't try to apply Gemma's filter.** The Gemma-tuned rule
   (low-norm + cap-no-space + AV mentions "research paper") catches almost
   nothing here. Qwen's failure surface is fundamentally different — no
   norm anomalies, no fixed templates, just a smooth gradient over
   (token-length, position).
