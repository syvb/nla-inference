# Deep token-level analysis — `results_20000.parquet`

N = **20000** activations.
Sink-token cluster (vec_norm ≥ 120,000): **11** rows (0.06 %). All cross-cuts below report on the **non-sink** subset unless noted.


## 1. Cos by token category (non-sink)

| cat | n | cos_mean | cos_median | cos_p10 | norm_mean |
|---|---|---|---|---|---|
| `cap_no_space` | 1054 | 0.976 | 0.995 | 0.986 | 68657 |
| `digit_only` | 954 | 0.990 | 0.991 | 0.983 | 57960 |
| `whitespace` | 1693 | 0.990 | 0.992 | 0.982 | 63041 |
| `subword_continuation` | 1888 | 0.991 | 0.993 | 0.986 | 65033 |
| `punct_only` | 4289 | 0.992 | 0.994 | 0.986 | 67904 |
| `single_char_alpha` | 1165 | 0.993 | 0.996 | 0.988 | 72295 |
| `cap_with_space` | 685 | 0.995 | 0.996 | 0.992 | 71891 |
| `word_lower_with_space` | 8236 | 0.996 | 0.997 | 0.995 | 80607 |

## 2. Cos by stripped token length (non-sink)

| len_bucket | n | cos_mean | cos_median | cos_p10 | norm_mean |
|---|---|---|---|---|---|
| `3` | 2655 | 0.992 | 0.997 | 0.991 | 76274 |
| `1` | 5467 | 0.992 | 0.994 | 0.986 | 67338 |
| `6-8` | 2547 | 0.993 | 0.996 | 0.990 | 75613 |
| `13-20` | 50 | 0.994 | 0.996 | 0.984 | 74165 |
| `2` | 2882 | 0.994 | 0.996 | 0.989 | 74354 |
| `4-5` | 3983 | 0.994 | 0.997 | 0.990 | 76226 |
| `9-12` | 712 | 0.995 | 0.997 | 0.992 | 78060 |

## 3. Cos by case × leading-space (alphabetic tokens, non-sink)

| case_ws | n | cos_mean | cos_median | cos_p10 | norm_mean |
|---|---|---|---|---|---|
| `no-space Cap` | 1267 | 0.978 | 0.995 | 0.986 | 68391 |
| `no-space lower` | 2266 | 0.991 | 0.993 | 0.986 | 65789 |
| `lead-space Cap` | 942 | 0.995 | 0.996 | 0.992 | 73889 |
| `lead-space lower` | 8553 | 0.996 | 0.997 | 0.994 | 80341 |

## 4. AV fallback templates — prevalence and quality

How often does the AV emit each known fallback phrase, and how does cos differ when it does vs doesn't?

| pattern | n_full | %_full | cos_when_present | cos_when_absent |
|---|---|---|---|---|
| `fragmented_academic` | 11 | 0.06 % | 1.000 | 0.993 |
| `tech_article_research_paper` | 15 | 0.07 % | 0.300 | 0.993 |
| `fibonacci` | 22 | 0.11 % | 0.610 | 0.993 |
| `tutorial_struct_research_paper` | 1 | 0.01 % | 0.112 | 0.993 |
| `kalman` | 7 | 0.04 % | 0.613 | 0.993 |
| `no_explanation_tags` | 2 | 0.01 % | 0.976 | 0.993 |

## 5. Worst individual samples per category (non-sink, cos < 0.7)

Total non-sink rows with cos < 0.7: **34** (0.17 % of non-sink).

Distribution by category:

| cat | n_bad | n_total_in_cat | bad_rate |
|---|---|---|---|
| cap_no_space | 28 | 1054 | 2.66 % |
| subword_continuation | 3 | 1888 | 0.16 % |
| single_char_alpha | 2 | 1165 | 0.17 % |
| punct_only | 1 | 4289 | 0.02 % |

### Examples (cos < 0.5, sorted by cos)

- cos=0.076  cat=`cap_no_space`  ‖v‖=36194  pos=254  tok=`'Android'`  context=`Android touchevent on view even with dialog in front
I am making video playback controls. …`
- cos=0.085  cat=`subword_continuation`  ‖v‖=36404  pos=123  tok=`'ubuntu'`  context=`ubuntu eclipse mars e(fx)clipse error on startup
I installed e(fx)clipse over my existing …`
- cos=0.101  cat=`cap_no_space`  ‖v‖=36550  pos=250  tok=`'Any'`  context=`Any suggestions for non-root access package management?
I've been a long time user of Ubun…`
- cos=0.106  cat=`cap_no_space`  ‖v‖=36150  pos=287  tok=`'VPN'`  context=`VPN Server freezing after client connection
I have a virtual Windows Server 2012 R2 and th…`
- cos=0.112  cat=`subword_continuation`  ‖v‖=36283  pos=342  tok=`'get'`  context=`getCookie using javascript

Possible Duplicate:
how to setCookie and getCookie using domai…`
- cos=0.143  cat=`cap_no_space`  ‖v‖=36411  pos=391  tok=`'How'`  context=`How to run .sql file using java without manually parsing the document
How to run .sql file…`
- cos=0.151  cat=`cap_no_space`  ‖v‖=36411  pos=369  tok=`'How'`  context=`How do I stop my function keys from acting as media controls?
Last week my function keys s…`
- cos=0.154  cat=`cap_no_space`  ‖v‖=36489  pos=232  tok=`'Changing'`  context=`Changing UINavigationBar height in UITableView
I am trying to change the height of the UIN…`
- cos=0.169  cat=`cap_no_space`  ‖v‖=36402  pos=258  tok=`'How'`  context=`How do I use the if statement to determen if the script should change the DNS-settings?
My…`
- cos=0.182  cat=`cap_no_space`  ‖v‖=36401  pos=326  tok=`'How'`  context=`How to stretch one object along multiple curves
How to stretch one object along multiple c…`
- cos=0.253  cat=`cap_no_space`  ‖v‖=36253  pos=34  tok=`'Device'`  context=`Device not tested during pre-launch Android
I've just uploaded an APK for an alpha release…`
- cos=0.262  cat=`cap_no_space`  ‖v‖=36505  pos=218  tok=`'Should'`  context=`Should i registerd all component to app.js or import in vue component
Now i have two optio…`
- cos=0.310  cat=`cap_no_space`  ‖v‖=36375  pos=102  tok=`'Up'`  context=`Upgrading from rails 5.2.4 to 6.1 with Delayed Job - Job's not being picked up
Working con…`
- cos=0.327  cat=`punct_only`  ‖v‖=36023  pos=337  tok=`'"'`  context=`"The Bonjour service could not be resolved."
After updating to 6.3 whenever I launch Xcode…`
- cos=0.339  cat=`cap_no_space`  ‖v‖=36233  pos=361  tok=`'Node'`  context=`NodeJS: Check that a string contains only GSM 03.38 characters
Given a string I want to ma…`
- cos=0.339  cat=`subword_continuation`  ‖v‖=36314  pos=354  tok=`'how'`  context=`how can I add icon for attachment in mesibo UI messagens in android SDK?
i'm trying to add…`
- cos=0.365  cat=`cap_no_space`  ‖v‖=36472  pos=269  tok=`'Distinct'`  context=`Distinct user can process unique data associated with them, then how to add this specific …`
- cos=0.377  cat=`cap_no_space`  ‖v‖=36304  pos=210  tok=`'Th'`  context=`ThriftColumnFamilyTemplate for querying super column family and their columns
I have a cas…`
- cos=0.398  cat=`single_char_alpha`  ‖v‖=36344  pos=186  tok=`'i'`  context=`i am not able to insatll my project pod getting this error
[!] Unable to find a specificat…`
- cos=0.399  cat=`single_char_alpha`  ‖v‖=36438  pos=325  tok=`'C'`  context=`C++ Builder 2009 - Form Layout Not Refreshing on Build
Using C++ Builder 2009, Win 7.
I ma…`
- cos=0.407  cat=`cap_no_space`  ‖v‖=36389  pos=364  tok=`'Time'`  context=`Time Series Regression Correlation

If I recall correctly, the equation for $R^2$ is $\fra…`
- cos=0.423  cat=`cap_no_space`  ‖v‖=36455  pos=217  tok=`'Unable'`  context=`Unable to resolve dependency in Android Studio
When starting Android Studio, I get this er…`
- cos=0.424  cat=`cap_no_space`  ‖v‖=36289  pos=270  tok=`'Why'`  context=`Why does Thunar suddenly see SD card as empty?
For several days, I have been able to acces…`
- cos=0.433  cat=`cap_no_space`  ‖v‖=36484  pos=435  tok=`'Implement'`  context=`Implement a page counter like the one in homescreen android
I am trying to implement the h…`
- cos=0.436  cat=`cap_no_space`  ‖v‖=36320  pos=252  tok=`'Self'`  context=`Self guided backcountry Monument Valley tour in a car like Outback?
I am planning to visit…`
- cos=0.456  cat=`cap_no_space`  ‖v‖=36403  pos=153  tok=`'How'`  context=`How to get formulas of multiple regressions by vectorizing
Suppose I have the following co…`
- cos=0.460  cat=`cap_no_space`  ‖v‖=36414  pos=313  tok=`'How'`  context=`How to efficiently add a 32 bit monotonically increasing contiguous id to each record in a…`
- cos=0.460  cat=`cap_no_space`  ‖v‖=36327  pos=113  tok=`'Show'`  context=`Show that $ζ$ is a Quadratic Integer in $Q[\sqrt{−3}]$
So in the complex plane, there are …`
- cos=0.460  cat=`cap_no_space`  ‖v‖=36294  pos=77  tok=`'Ps'`  context=`Psycopg2 Full Outer Join with Python 2.6
I am brand new to using psycopg2 to interact with…`
- cos=0.470  cat=`cap_no_space`  ‖v‖=36292  pos=128  tok=`'Asp'`  context=`Asp.net MVC 4 redirect when user needs to update information
On any action I add code on o…`
- cos=0.472  cat=`cap_no_space`  ‖v‖=36421  pos=390  tok=`'How'`  context=`How to run Cucumber test in NodeJS which is written in Typescript and ES6
What is the prop…`
- cos=0.495  cat=`cap_no_space`  ‖v‖=36287  pos=151  tok=`'Using'`  context=`Using Fabric to start Locust on multiple slaves
I am rather new to Fabric but I started wo…`

## 6. Sink-token cluster (vec_norm ≥ 120,000)

N = **11** rows. Top-1 fallback phrase prevalence:

| pattern | %_in_sink | %_in_main |
|---|---|---|
| `fragmented_academic` | 100.0 % | 0.0 % |
| `tech_article_research_paper` | 0.0 % | 0.1 % |
| `fibonacci` | 0.0 % | 0.1 % |
| `tutorial_struct_research_paper` | 0.0 % | 0.0 % |
| `kalman` | 0.0 % | 0.0 % |
| `no_explanation_tags` | 0.0 % | 0.0 % |

Mean cos in sink cluster: **1.000** vs **0.993** in non-sink.

Top-10 sink tokens by frequency:

| token | n |
|---|---|
| `' '` | 8 |
| `"'"` | 3 |

## 7. Cos vs AV explanation length (non-sink)


## 7a. Cos by explanation length (words)

| expl_word_bucket | n | cos_mean | cos_median | cos_p10 | norm_mean |
|---|---|---|---|---|---|
| `51-70` | 300 | 0.988 | 0.989 | 0.982 | 57389 |
| `71-100` | 14126 | 0.992 | 0.995 | 0.987 | 69815 |
| `100+` | 5563 | 0.994 | 0.997 | 0.994 | 79761 |

## 8. Position effect (non-sink, decile bins)


## 8. Cos by position decile (non-sink)

| pos_bucket | n | cos_mean | cos_median | cos_p10 | norm_mean |
|---|---|---|---|---|---|
| `(219.0, 273.0]` | 2025 | 0.991 | 0.995 | 0.987 | 71219 |
| `(321.0, 364.0]` | 1992 | 0.991 | 0.996 | 0.987 | 72257 |
| `(91.0, 161.0]` | 2013 | 0.992 | 0.995 | 0.987 | 71299 |
| `(161.0, 219.0]` | 1980 | 0.992 | 0.995 | 0.987 | 71078 |
| `(273.0, 321.0]` | 1998 | 0.993 | 0.995 | 0.987 | 71532 |
| `(364.0, 404.0]` | 2008 | 0.993 | 0.996 | 0.988 | 72792 |
| `(404.0, 440.0]` | 1977 | 0.994 | 0.996 | 0.989 | 73190 |
| `(9.999, 91.0]` | 2010 | 0.994 | 0.996 | 0.990 | 73776 |
| `(440.0, 477.0]` | 2022 | 0.994 | 0.996 | 0.988 | 73570 |
| `(477.0, 511.0]` | 1964 | 0.994 | 0.996 | 0.989 | 73265 |

## 9. Single-character alphabetic tokens — which letters reconstruct worst?

Bottom-20 single-character tokens (n ≥ 3):
| token | n | cos_mean |
|---|---|---|
| `'C'` | 10 | 0.930 |
| `'i'` | 19 | 0.957 |
| `'z'` | 5 | 0.986 |
| `'O'` | 4 | 0.988 |
| `' n'` | 6 | 0.988 |
| `'q'` | 3 | 0.988 |
| `'y'` | 9 | 0.988 |
| `'n'` | 16 | 0.989 |
| `'j'` | 4 | 0.989 |
| `'D'` | 6 | 0.989 |
| `'b'` | 13 | 0.990 |
| `'r'` | 6 | 0.990 |
| `'A'` | 10 | 0.990 |
| `'c'` | 13 | 0.990 |

## 10. AV explanations seen multiple times verbatim (≥ 5 copies)

If the AV emits identical text for many distinct activations, that's a memorised template — likely a fallback when the AV can't extract content.

| n_copies | mean_cos_when_emitted | first 200 chars of explanation |
|---|---|---|
| 9 | 1.000 | `Fragmented, incoherent academic text pattern: fragmented sentences and garbled phrasing suggest a collage or corrupted document, with fragmented literary/cultural references.  The phrase "it'" signals` |
| 7 | 0.237 | `Technical article structure: a blog post reviewing a specific research paper, with a consistent pattern of explaining the problem and solution.  The phrase "The results of the study are:" signals a st` |