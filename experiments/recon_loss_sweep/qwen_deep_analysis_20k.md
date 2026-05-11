# Deep token-level analysis — `results_qwen7_20000.parquet`

N = **20000** activations.
Sink-token cluster (vec_norm ≥ 200): **0** rows (0.00 %). All cross-cuts below report on the **non-sink** subset unless noted.


## 1. Cos by token category (non-sink)

| cat | n | cos_mean | cos_median | cos_p10 | norm_mean |
|---|---|---|---|---|---|
| `digit_only` | 1032 | 0.839 | 0.842 | 0.768 | 107 |
| `whitespace` | 968 | 0.841 | 0.848 | 0.771 | 106 |
| `code_syntax` | 296 | 0.848 | 0.855 | 0.780 | 113 |
| `punct_only` | 3713 | 0.854 | 0.863 | 0.788 | 111 |
| `subword_continuation` | 1419 | 0.860 | 0.867 | 0.795 | 116 |
| `other` | 805 | 0.866 | 0.871 | 0.806 | 117 |
| `cap_no_space` | 953 | 0.871 | 0.877 | 0.806 | 117 |
| `single_char_alpha` | 1028 | 0.878 | 0.892 | 0.801 | 116 |
| `cap_with_space` | 830 | 0.891 | 0.898 | 0.841 | 120 |
| `word_lower_with_space` | 8937 | 0.896 | 0.901 | 0.851 | 121 |

## 2. Cos by stripped token length (non-sink)

| len_bucket | n | cos_mean | cos_median | cos_p10 | norm_mean |
|---|---|---|---|---|---|
| `1` | 4812 | 0.858 | 0.867 | 0.787 | 112 |
| `13-20` | 58 | 0.876 | 0.889 | 0.810 | 123 |
| `2` | 3320 | 0.877 | 0.887 | 0.809 | 116 |
| `4-5` | 4240 | 0.885 | 0.892 | 0.830 | 120 |
| `3` | 2994 | 0.886 | 0.895 | 0.830 | 117 |
| `6-8` | 2887 | 0.888 | 0.894 | 0.833 | 122 |
| `9-12` | 718 | 0.890 | 0.895 | 0.837 | 124 |

## 3. Cos by case × leading-space (alphabetic tokens, non-sink)

| case_ws | n | cos_mean | cos_median | cos_p10 | norm_mean |
|---|---|---|---|---|---|
| `no-space lower` | 1625 | 0.857 | 0.865 | 0.790 | 116 |
| `no-space Cap` | 1157 | 0.870 | 0.877 | 0.804 | 116 |
| `lead-space Cap` | 1096 | 0.893 | 0.900 | 0.842 | 120 |
| `lead-space lower` | 9289 | 0.896 | 0.901 | 0.850 | 121 |

## 4. AV fallback templates — prevalence and quality

How often does the AV emit each known fallback phrase, and how does cos differ when it does vs doesn't?

| pattern | n_full | %_full | cos_when_present | cos_when_absent |
|---|---|---|---|---|
| `fragmented_academic` | 0 | 0.00 % | nan | 0.876 |
| `tech_article_research_paper` | 0 | 0.00 % | nan | 0.876 |
| `fibonacci` | 10 | 0.05 % | 0.851 | 0.876 |
| `tutorial_struct_research_paper` | 0 | 0.00 % | nan | 0.876 |
| `kalman` | 5 | 0.03 % | 0.879 | 0.876 |
| `no_explanation_tags` | 2 | 0.01 % | 0.612 | 0.876 |

## 5. Worst individual samples per category (non-sink, cos < 0.7)

Total non-sink rows with cos < 0.7: **101** (0.51 % of non-sink).

Distribution by category:

| cat | n_bad | n_total_in_cat | bad_rate |
|---|---|---|---|
| punct_only | 29 | 3713 | 0.78 % |
| digit_only | 21 | 1032 | 2.03 % |
| whitespace | 11 | 968 | 1.14 % |
| subword_continuation | 11 | 1419 | 0.78 % |
| single_char_alpha | 9 | 1028 | 0.88 % |
| word_lower_with_space | 8 | 8937 | 0.09 % |
| code_syntax | 5 | 296 | 1.69 % |
| cap_with_space | 3 | 830 | 0.36 % |
| other | 3 | 805 | 0.37 % |
| cap_no_space | 1 | 953 | 0.10 % |

### Examples (cos < 0.5, sorted by cos)

- cos=0.368  cat=`digit_only`  ‖v‖=112  pos=123  tok=`'7'`  context=`Jsoup crawler and HTTP error fetching URL
I am writing a crawler with Jsoup and this is th…`
- cos=0.481  cat=`punct_only`  ‖v‖=122  pos=491  tok=`"''"`  context=`Is there a way to take a screenshot everytime the key ''Space'' is pressed and released?
I…`
- cos=0.490  cat=`subword_continuation`  ‖v‖=124  pos=118  tok=`'ab'`  context=`CSS navigation hover dropdown speed?
I have two problems regarding my code:

Navigation me…`
- cos=0.500  cat=`word_lower_with_space`  ‖v‖=137  pos=505  tok=`' stub'`  context=`Problems with creating a new/next Android activity
I just wanted to click on a button in t…`

## 6. Sink-token cluster (vec_norm ≥ 200)

N = **0** rows. Top-1 fallback phrase prevalence:

| pattern | %_in_sink | %_in_main |
|---|---|---|
| `fragmented_academic` | 0.0 % | 0.0 % |
| `tech_article_research_paper` | 0.0 % | 0.0 % |
| `fibonacci` | 0.0 % | 0.1 % |
| `tutorial_struct_research_paper` | 0.0 % | 0.0 % |
| `kalman` | 0.0 % | 0.0 % |
| `no_explanation_tags` | 0.0 % | 0.0 % |

Mean cos in sink cluster: **nan** vs **0.876** in non-sink.

Top-10 sink tokens by frequency:

| token | n |
|---|---|

## 7. Cos vs AV explanation length (non-sink)


## 7a. Cos by explanation length (words)

| expl_word_bucket | n | cos_mean | cos_median | cos_p10 | norm_mean |
|---|---|---|---|---|---|
| `51-70` | 104 | 0.810 | 0.823 | 0.739 | 110 |
| `71-100` | 9323 | 0.858 | 0.866 | 0.789 | 114 |
| `100+` | 10573 | 0.892 | 0.899 | 0.843 | 119 |

## 8. Position effect (non-sink, decile bins)


## 8. Cos by position decile (non-sink)

| pos_bucket | n | cos_mean | cos_median | cos_p10 | norm_mean |
|---|---|---|---|---|---|
| `(423.0, 511.0]` | 1995 | 0.860 | 0.870 | 0.794 | 114 |
| `(349.0, 423.0]` | 2002 | 0.860 | 0.870 | 0.793 | 113 |
| `(290.0, 349.0]` | 1971 | 0.865 | 0.874 | 0.797 | 114 |
| `(234.0, 290.0]` | 2006 | 0.867 | 0.876 | 0.801 | 114 |
| `(189.0, 234.0]` | 1984 | 0.869 | 0.879 | 0.803 | 115 |
| `(148.0, 189.0]` | 2001 | 0.873 | 0.883 | 0.810 | 116 |
| `(111.0, 148.0]` | 2002 | 0.878 | 0.886 | 0.820 | 117 |
| `(75.0, 111.0]` | 2031 | 0.886 | 0.894 | 0.830 | 118 |
| `(42.0, 75.0]` | 2004 | 0.893 | 0.902 | 0.838 | 120 |
| `(9.999, 42.0]` | 2004 | 0.906 | 0.915 | 0.860 | 124 |

## 9. Single-character alphabetic tokens — which letters reconstruct worst?

Bottom-20 single-character tokens (n ≥ 3):
| token | n | cos_mean |
|---|---|---|
| `'h'` | 5 | 0.766 |
| `'q'` | 9 | 0.770 |
| `'R'` | 3 | 0.797 |
| `'u'` | 3 | 0.797 |
| `'b'` | 6 | 0.810 |
| `'r'` | 3 | 0.814 |
| `'f'` | 13 | 0.815 |
| `'Z'` | 5 | 0.816 |
| `'g'` | 7 | 0.819 |
| `' j'` | 4 | 0.820 |
| `'c'` | 8 | 0.821 |
| `'i'` | 6 | 0.822 |

## 10. AV explanations seen multiple times verbatim (≥ 5 copies)

If the AV emits identical text for many distinct activations, that's a memorised template — likely a fallback when the AV can't extract content.

| n_copies | mean_cos_when_emitted | first 200 chars of explanation |
|---|---|---|