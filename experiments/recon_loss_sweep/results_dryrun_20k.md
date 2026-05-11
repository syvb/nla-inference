# Reconstruction-loss sweep — `results_20000.parquet`

**N = 20000** activations.
AV parse rate: **1.000** (19998 / 20000)


## Overall MSE / cos

| metric | mean | p10 | p50 | p90 | p99 |
|---|---|---|---|---|---|
| cos | 0.993 | 0.988 | 0.996 | 0.998 | 0.998 |
| mse_nrm | 0.014 | 0.005 | 0.009 | 0.024 | 0.047 |

## Top 30 BEST-reconstructed tokens (highest cos, n ≥ 5)

| rank | token | n | cos_mean | mse_mean | norm_mean |
|---|---|---|---|---|---|
| 1 | ` been` | 5 | 0.998 | 0.004 | 87935.1 |
| 2 | ` large` | 5 | 0.998 | 0.004 | 86576.2 |
| 3 | ` connect` | 5 | 0.998 | 0.004 | 89391.9 |
| 4 | ` best` | 5 | 0.998 | 0.004 | 86344.1 |
| 5 | ` always` | 6 | 0.998 | 0.004 | 90734.0 |
| 6 | ` seem` | 6 | 0.998 | 0.004 | 80695.3 |
| 7 | ` works` | 13 | 0.998 | 0.004 | 90537.1 |
| 8 | ` specific` | 8 | 0.998 | 0.004 | 90549.4 |
| 9 | ` provide` | 5 | 0.998 | 0.004 | 90122.9 |
| 10 | ` good` | 8 | 0.998 | 0.005 | 88383.6 |
| 11 | ` sure` | 7 | 0.998 | 0.005 | 81605.1 |
| 12 | ` look` | 9 | 0.998 | 0.005 | 84897.8 |
| 13 | ` could` | 14 | 0.998 | 0.005 | 90374.4 |
| 14 | ` might` | 7 | 0.998 | 0.005 | 90606.7 |
| 15 | ` much` | 5 | 0.998 | 0.005 | 82369.9 |
| 16 | ` wrong` | 5 | 0.998 | 0.005 | 85518.1 |
| 17 | ` following` | 16 | 0.998 | 0.005 | 90140.8 |
| 18 | ` seems` | 10 | 0.998 | 0.005 | 88649.1 |
| 19 | ` input` | 5 | 0.998 | 0.005 | 82475.1 |
| 20 | ` will` | 26 | 0.998 | 0.005 | 89514.4 |
| 21 | ` between` | 5 | 0.998 | 0.005 | 83821.3 |
| 22 | ` any` | 29 | 0.998 | 0.005 | 86888.1 |
| 23 | ` fix` | 7 | 0.997 | 0.005 | 81369.7 |
| 24 | `ll` | 7 | 0.997 | 0.005 | 84408.4 |
| 25 | ` working` | 6 | 0.997 | 0.005 | 82761.0 |
| 26 | ` But` | 11 | 0.997 | 0.005 | 93958.7 |
| 27 | ` tried` | 11 | 0.997 | 0.005 | 86661.8 |
| 28 | ` may` | 8 | 0.997 | 0.005 | 90941.3 |
| 29 | ` without` | 16 | 0.997 | 0.005 | 94305.2 |
| 30 | ` trying` | 15 | 0.997 | 0.005 | 79029.7 |

## Bottom 30 WORST-reconstructed tokens (lowest cos, n ≥ 5)

| rank | token | n | cos_mean | mse_mean | norm_mean |
|---|---|---|---|---|---|
| 1 | `How` | 13 | 0.616 | 0.767 | 55489.7 |
| 2 | `get` | 5 | 0.816 | 0.367 | 63946.5 |
| 3 | `how` | 6 | 0.886 | 0.229 | 63563.2 |
| 4 | `C` | 10 | 0.930 | 0.140 | 54839.5 |
| 5 | `What` | 12 | 0.957 | 0.086 | 77111.3 |
| 6 | `i` | 19 | 0.957 | 0.086 | 57208.4 |
| 7 | `�` | 5 | 0.980 | 0.039 | 51624.1 |
| 8 | `"` | 71 | 0.982 | 0.036 | 70393.2 |
| 9 | `},` | 5 | 0.982 | 0.035 | 61746.7 |
| 10 | `               ` | 5 | 0.984 | 0.031 | 60208.3 |
| 11 | `");` | 5 | 0.985 | 0.030 | 61545.6 |
| 12 | `}\\` | 5 | 0.985 | 0.030 | 64809.6 |
| 13 | `                        ` | 7 | 0.985 | 0.030 | 60070.1 |
| 14 | `}{` | 5 | 0.985 | 0.029 | 71195.1 |
| 15 | `                    ` | 5 | 0.986 | 0.029 | 63943.6 |
| 16 | `String` | 6 | 0.986 | 0.029 | 66189.6 |
| 17 | ` |` | 6 | 0.986 | 0.028 | 63826.3 |
| 18 | `z` | 5 | 0.986 | 0.028 | 57918.7 |
| 19 | `}` | 49 | 0.986 | 0.027 | 57193.6 |
| 20 | `",` | 17 | 0.987 | 0.026 | 60421.7 |
| 21 | `});` | 7 | 0.987 | 0.026 | 57535.8 |
| 22 | `      ` | 15 | 0.987 | 0.026 | 60544.1 |
| 23 | `ADDRESS` | 7 | 0.987 | 0.026 | 57189.7 |
| 24 | `:\\` | 5 | 0.987 | 0.026 | 69937.1 |
| 25 | `9` | 58 | 0.987 | 0.026 | 54399.0 |
| 26 | `);` | 30 | 0.987 | 0.025 | 54912.6 |
| 27 | `script` | 8 | 0.987 | 0.025 | 61091.0 |
| 28 | `();` | 14 | 0.988 | 0.025 | 57909.1 |
| 29 | `path` | 6 | 0.988 | 0.025 | 59399.1 |
| 30 | ` }` | 6 | 0.988 | 0.024 | 60906.4 |

## Cos by activation L2-norm bucket (decile)

| bucket | norm_range | n | cos_mean | cos_median |
|---|---|---|---|---|
| 0 | 25919 – 54293 | 2000 | 0.978 | 0.990 |
| 1 | 54293 – 59819 | 2000 | 0.990 | 0.992 |
| 2 | 59819 – 64243 | 2000 | 0.991 | 0.993 |
| 3 | 64243 – 68577 | 2000 | 0.993 | 0.994 |
| 4 | 68577 – 72835 | 2000 | 0.994 | 0.995 |
| 5 | 72835 – 76847 | 2000 | 0.995 | 0.996 |
| 6 | 76847 – 80924 | 2000 | 0.996 | 0.997 |
| 7 | 80924 – 85108 | 2000 | 0.997 | 0.997 |
| 8 | 85108 – 90195 | 2000 | 0.997 | 0.997 |
| 9 | 90195 – 477188 | 2000 | 0.997 | 0.998 |

## Cos by token position bucket (decile, all positions ≥ 10)

| bucket | pos_range | n | cos_mean |
|---|---|---|---|
| 0 | 10 – 91 | 1974 | 0.994 |
| 1 | 91 – 161 | 2019 | 0.992 |
| 2 | 161 – 219 | 1964 | 0.992 |
| 3 | 219 – 273 | 2030 | 0.991 |
| 4 | 273 – 321 | 1987 | 0.993 |
| 5 | 321 – 364 | 2007 | 0.992 |
| 6 | 364 – 404 | 2000 | 0.993 |
| 7 | 404 – 440 | 1967 | 0.994 |
| 8 | 440 – 477 | 2037 | 0.994 |
| 9 | 477 – 511 | 2015 | 0.994 |

## Cos by sequence length bucket (decile)

| bucket | seq_len_range | n | cos_mean |
|---|---|---|---|
| 0 | 22 – 216 | 1994 | 0.991 |
| 1 | 216 – 303 | 2003 | 0.991 |
| 2 | 303 – 385 | 1979 | 0.993 |
| 3 | 385 – 472 | 2022 | 0.993 |
| 4 | 472 – 512 | 872 | 0.993 |
| 5 | 512 – 512 | 0 | nan |
| 6 | 512 – 512 | 0 | nan |
| 7 | 512 – 512 | 0 | nan |
| 8 | 512 – 512 | 0 | nan |
| 9 | 512 – 512 | 11130 | 0.993 |

## Qualitative — 10 best-reconstructed individual samples

- cos=1.000 pos=344 ‖v‖=448517.6 tok=` `  expl=`Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a collage or corrupted document excerpt from a literary/cultural`
- cos=1.000 pos=495 ‖v‖=468996.1 tok=` `  expl=`Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a collage or corrupted document, with fragmented literary/cultur`
- cos=1.000 pos=495 ‖v‖=477188.1 tok=` `  expl=`Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a text-generation or scraped/fragmented passage about a literary`
- cos=1.000 pos=425 ‖v‖=444421.3 tok=`'`  expl=`Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a collage or corrupted document, with fragmented literary/cultur`
- cos=1.000 pos=422 ‖v‖=432133.5 tok=` `  expl=`Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a collage or corrupted document, with fragmented literary/cultur`
- cos=1.000 pos=475 ‖v‖=460804.6 tok=`'`  expl=`Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a collage or corrupted document, with fragmented literary/cultur`
- cos=1.000 pos=411 ‖v‖=446469.0 tok=` `  expl=`Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a collage or corrupted document, with fragmented literary/cultur`
- cos=1.000 pos=483 ‖v‖=464900.3 tok=` `  expl=`Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a collage or corrupted document, with fragmented literary/cultur`
- cos=1.000 pos=479 ‖v‖=442373.6 tok=` `  expl=`Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a collage or corrupted document, with fragmented literary/cultur`
- cos=1.000 pos=415 ‖v‖=466948.5 tok=` `  expl=`Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a collage or corrupted document, with fragmented literary/cultur`

## Qualitative — 10 worst-reconstructed individual samples

- cos=0.076 pos=254 ‖v‖=36193.5 tok=`Android`  expl=`Technical article structure: a blog post reviewing a specific research paper, with a structured argument about the Fibonacci sequence and the 3D model.

The phr`
- cos=0.085 pos=123 ‖v‖=36404.0 tok=`ubuntu`  expl=`Technical tutorial structure: article is presenting a detailed explanation of a specific programming topic, with a structured argument about the Kalman filter.
`
- cos=0.101 pos=250 ‖v‖=36549.8 tok=`Any`  expl=`Technical article structure: a blog post reviewing a specific research paper, with a consistent pattern of explaining the problem and solution.

The phrase "The`
- cos=0.106 pos=287 ‖v‖=36149.7 tok=`VPN`  expl=`Technical article structure: informational blog post with structured argument, following a pattern of explaining a specific research topic with citations.

The `
- cos=0.112 pos=342 ‖v‖=36282.5 tok=`get`  expl=`Technical tutorial structure: article is presenting a detailed explanation of a specific research paper, with a list of topics covering the Fibonacci sequence.
`
- cos=0.143 pos=391 ‖v‖=36410.6 tok=`How`  expl=`Technical article structure: a blog post reviewing a specific research paper, with a consistent pattern of explaining the problem and solution.

The phrase "The`
- cos=0.151 pos=369 ‖v‖=36410.9 tok=`How`  expl=`Technical article structure: a blog post reviewing a specific research paper, with a structured argument about the Fibonacci sequence and the Riemann hypothesis`
- cos=0.154 pos=232 ‖v‖=36489.0 tok=`Changing`  expl=`Technical article structure: informational blog post with structured argument, following a pattern of explaining a specific research topic with citations.

The `
- cos=0.169 pos=258 ‖v‖=36402.4 tok=`How`  expl=`Technical article structure: a blog post reviewing a specific research paper, with a consistent pattern of explaining the problem and solution.

The phrase "The`
- cos=0.182 pos=326 ‖v‖=36400.8 tok=`How`  expl=`Technical article structure: educational blog post with structured argumentation, following a pattern of explaining a specific topic with citations.

The phrase`

## Qualitative — 10 random samples

- cos=0.997 pos=441 ‖v‖=76760.2 tok=` is`  expl=`Technical Q&A forum post establishing context: user is asking about a specific tooling/language problem involving C#/.NET/FFI.

The phrase "There is" signals a `
- cos=0.996 pos=454 ‖v‖=51982.1 tok=` `  expl=`Technical tutorial format: instructional post about Silverlight/WPF, with a specific bug description and workaround for Silverlight 3.

The phrase "Silverlight `
- cos=0.996 pos=309 ‖v‖=65382.6 tok=`proxy`  expl=`Man-page documentation format: structured listing of Qt/KDE library dependencies, with package names and descriptions following a consistent pattern.

The phras`
- cos=0.994 pos=449 ‖v‖=65953.1 tok=`Session`  expl=`Technical documentation pattern: ASP.NET/webforms context, with structured code block showing HTTP cookie/session state example.

The phrase "HttpContext.Curren`
- cos=0.997 pos=459 ‖v‖=95991.6 tok=`You`  expl=`Technical Q&A format: answer structure expects a fix/explanation for a Polish/Qt bug, with a code review or debugging advice.

The phrase "The problem is that y`
- cos=0.997 pos=46 ‖v‖=62215.7 tok=`'`  expl=`Technical GIS instructional format: structured tutorial with numbered steps, describing a spatial analysis workflow for ArcGIS Pro.

The phrase "In ArcGIS Pro, `
- cos=0.998 pos=195 ‖v‖=83694.5 tok=` held`  expl=`Technical documentation describing Windows Update client behavior, with a KB article format explaining policy settings and driver release timing.

The sentence `
- cos=0.994 pos=186 ‖v‖=76543.8 tok=`(`  expl=`Tutorial structure: documentation is explaining a module/views configuration, with a French/Drupal UI example showing path alias and node ID.

The sentence "exe`
- cos=0.997 pos=282 ‖v‖=87549.8 tok=` for`  expl=`Technical documentation structure: description of a formula/algorithm, expecting a solution/example for Russian database table creation.

The phrase "Example fo`
- cos=0.994 pos=71 ‖v‖=50664.2 tok=`2`  expl=`Technical tutorial structure: forum post format with code examples, requiring a list of variables/parameters for a dashboard UI.

The pattern "var p1 = 1; var p`