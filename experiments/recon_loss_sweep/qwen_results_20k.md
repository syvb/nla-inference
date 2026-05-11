# Reconstruction-loss sweep — `results_qwen7_20000.parquet`

**N = 20000** activations.
AV parse rate: **1.000** (19998 / 20000)


## Overall MSE / cos

| metric | mean | p10 | p50 | p90 | p99 |
|---|---|---|---|---|---|
| cos | 0.876 | 0.810 | 0.885 | 0.928 | 0.949 |
| mse_nrm | 0.249 | 0.143 | 0.229 | 0.379 | 0.552 |

## Top 30 BEST-reconstructed tokens (highest cos, n ≥ 5)

| rank | token | n | cos_mean | mse_mean | norm_mean |
|---|---|---|---|---|---|
| 1 | ` tried` | 9 | 0.933 | 0.135 | 129.8 |
| 2 | ` API` | 5 | 0.931 | 0.138 | 131.2 |
| 3 | ` another` | 5 | 0.929 | 0.142 | 121.8 |
| 4 | ` works` | 6 | 0.929 | 0.142 | 136.7 |
| 5 | ` own` | 6 | 0.928 | 0.143 | 118.6 |
| 6 | ` issue` | 7 | 0.928 | 0.144 | 137.1 |
| 7 | ` help` | 22 | 0.927 | 0.146 | 137.7 |
| 8 | ` program` | 6 | 0.927 | 0.146 | 130.4 |
| 9 | ` web` | 5 | 0.925 | 0.150 | 119.1 |
| 10 | ` without` | 7 | 0.925 | 0.151 | 123.5 |
| 11 | ` good` | 5 | 0.924 | 0.151 | 132.5 |
| 12 | `When` | 7 | 0.924 | 0.152 | 121.3 |
| 13 | ` working` | 6 | 0.922 | 0.156 | 134.0 |
| 14 | ` perfectly` | 5 | 0.922 | 0.157 | 132.8 |
| 15 | ` getting` | 6 | 0.922 | 0.157 | 125.9 |
| 16 | `'ve` | 12 | 0.921 | 0.158 | 132.6 |
| 17 | ` number` | 6 | 0.920 | 0.160 | 120.8 |
| 18 | ` idea` | 6 | 0.920 | 0.160 | 136.6 |
| 19 | ` having` | 7 | 0.919 | 0.162 | 128.7 |
| 20 | ` running` | 6 | 0.918 | 0.163 | 127.8 |
| 21 | ` problem` | 17 | 0.918 | 0.164 | 134.1 |
| 22 | ` approach` | 5 | 0.918 | 0.165 | 126.4 |
| 23 | ` some` | 19 | 0.917 | 0.166 | 122.3 |
| 24 | `'m` | 32 | 0.917 | 0.166 | 131.5 |
| 25 | ` application` | 8 | 0.917 | 0.166 | 129.6 |
| 26 | ` give` | 8 | 0.917 | 0.167 | 129.0 |
| 27 | ` following` | 15 | 0.916 | 0.167 | 126.4 |
| 28 | ` cannot` | 9 | 0.916 | 0.167 | 127.6 |
| 29 | ` sending` | 5 | 0.916 | 0.168 | 121.0 |
| 30 | ` create` | 10 | 0.916 | 0.168 | 120.6 |

## Bottom 30 WORST-reconstructed tokens (lowest cos, n ≥ 5)

| rank | token | n | cos_mean | mse_mean | norm_mean |
|---|---|---|---|---|---|
| 1 | `h` | 5 | 0.766 | 0.467 | 107.6 |
| 2 | `q` | 9 | 0.770 | 0.460 | 106.2 |
| 3 | `}(` | 6 | 0.791 | 0.419 | 106.6 |
| 4 | `}` | 17 | 0.791 | 0.417 | 106.7 |
| 5 | `(x` | 6 | 0.795 | 0.410 | 106.1 |
| 6 | ` )` | 7 | 0.800 | 0.399 | 113.3 |
| 7 | `]` | 14 | 0.804 | 0.393 | 108.0 |
| 8 | `math` | 7 | 0.804 | 0.393 | 91.5 |
| 9 | `\n\n\n` | 6 | 0.807 | 0.387 | 108.4 |
| 10 | `�` | 7 | 0.807 | 0.387 | 107.4 |
| 11 | ` |` | 7 | 0.807 | 0.385 | 112.3 |
| 12 | `^` | 9 | 0.808 | 0.384 | 108.0 |
| 13 | `IP` | 5 | 0.809 | 0.381 | 106.4 |
| 14 | `b` | 6 | 0.810 | 0.381 | 109.0 |
| 15 | `  ` | 22 | 0.810 | 0.379 | 104.2 |
| 16 | `"\n` | 17 | 0.813 | 0.373 | 99.6 |
| 17 | `_{` | 6 | 0.814 | 0.372 | 105.2 |
| 18 | `f` | 13 | 0.815 | 0.371 | 110.3 |
| 19 | `bb` | 6 | 0.815 | 0.370 | 103.2 |
| 20 | `':` | 6 | 0.815 | 0.370 | 109.6 |
| 21 | `{` | 23 | 0.815 | 0.369 | 104.7 |
| 22 | `_` | 15 | 0.815 | 0.369 | 97.1 |
| 23 | `Z` | 5 | 0.816 | 0.367 | 106.8 |
| 24 | `div` | 10 | 0.818 | 0.364 | 110.8 |
| 25 | `>` | 19 | 0.819 | 0.363 | 109.0 |
| 26 | `g` | 7 | 0.819 | 0.363 | 107.9 |
| 27 | ` @` | 8 | 0.821 | 0.359 | 110.4 |
| 28 | `>\n` | 26 | 0.821 | 0.358 | 98.1 |
| 29 | `c` | 8 | 0.821 | 0.358 | 108.7 |
| 30 | `\\` | 39 | 0.822 | 0.357 | 101.8 |

## Cos by activation L2-norm bucket (decile)

| bucket | norm_range | n | cos_mean | cos_median |
|---|---|---|---|---|
| 0 | 74 – 102 | 2000 | 0.826 | 0.830 |
| 1 | 102 – 107 | 2000 | 0.849 | 0.855 |
| 2 | 107 – 111 | 2000 | 0.862 | 0.868 |
| 3 | 111 – 114 | 2000 | 0.869 | 0.876 |
| 4 | 114 – 117 | 2000 | 0.877 | 0.885 |
| 5 | 117 – 120 | 2000 | 0.883 | 0.890 |
| 6 | 120 – 123 | 2000 | 0.887 | 0.894 |
| 7 | 123 – 126 | 2000 | 0.894 | 0.901 |
| 8 | 126 – 131 | 2000 | 0.901 | 0.908 |
| 9 | 131 – 162 | 2000 | 0.911 | 0.920 |

## Cos by token position bucket (decile, all positions ≥ 10)

| bucket | pos_range | n | cos_mean |
|---|---|---|---|
| 0 | 10 – 42 | 1945 | 0.906 |
| 1 | 42 – 75 | 2001 | 0.893 |
| 2 | 75 – 111 | 2047 | 0.886 |
| 3 | 111 – 148 | 1993 | 0.878 |
| 4 | 148 – 189 | 2013 | 0.874 |
| 5 | 189 – 234 | 1998 | 0.869 |
| 6 | 234 – 290 | 1996 | 0.867 |
| 7 | 290 – 349 | 1980 | 0.865 |
| 8 | 349 – 423 | 2004 | 0.860 |
| 9 | 423 – 511 | 2023 | 0.860 |

## Cos by sequence length bucket (decile)

| bucket | seq_len_range | n | cos_mean |
|---|---|---|---|
| 0 | 20 – 201 | 1997 | 0.895 |
| 1 | 201 – 281 | 1981 | 0.887 |
| 2 | 281 – 358 | 2016 | 0.882 |
| 3 | 358 – 435 | 1985 | 0.875 |
| 4 | 435 – 512 | 1770 | 0.877 |
| 5 | 512 – 512 | 0 | nan |
| 6 | 512 – 512 | 0 | nan |
| 7 | 512 – 512 | 0 | nan |
| 8 | 512 – 512 | 0 | nan |
| 9 | 512 – 512 | 10251 | 0.868 |

## Qualitative — 10 best-reconstructed individual samples

- cos=0.968 pos=322 ‖v‖=101.3 tok=` Ky`  expl=`Technical paper format with Greek engineering journal article structure, listing simulation results and performance metrics for MATLAB's control algorithms in s`
- cos=0.967 pos=220 ‖v‖=104.6 tok=` Howard`  expl=`Technical MSDN blog post format with code examples showing async/await in WPF, discussing a UI performance issue with `System.IO`.

The sentence "I've seen a fe`
- cos=0.966 pos=151 ‖v‖=107.9 tok=` Mut`  expl=`Technical blog post format with informal tone discussing Ethereum's Solidity language, listing historical context about contract languages and their execution.
`
- cos=0.965 pos=21 ‖v‖=143.4 tok=` class`  expl=`Indian Android/Java developer asking about AsyncTask and UI thread access, implying code snippet or question about handling multiple threads and data from a ser`
- cos=0.964 pos=80 ‖v‖=145.8 tok=` suggestions`  expl=`Technical/Engineering forum post asking about motor failure in a PCB design context, implying a user is experiencing issues with a specific component or system `
- cos=0.963 pos=182 ‖v‖=110.8 tok=` Key`  expl=`Indian IT/Security expert blog post with technical explanation around OAuth, JWT, and SAML for web apps, mixing technical details with UI/UX context.

The sente`
- cos=0.963 pos=54 ‖v‖=147.0 tok=` provide`  expl=`Malaysian/PHP developer asking for CSS/HTML layout help with a specific div structure issue around responsive design and positioning of a form.

The phrase "So `
- cos=0.963 pos=10 ‖v‖=132.7 tok=` REST`  expl=`Technical/IT support context with a question about pulling Salesforce data via REST API, implying a developer or business user seeking data access methodology.
`
- cos=0.962 pos=400 ‖v‖=106.2 tok=`ball`  expl=`Code snippet with Python/SQL context, showing a regex expression to filter pandas DataFrame rows with "NaN" values, suggesting a Stack Overflow or similar post.`
- cos=0.962 pos=45 ‖v‖=139.8 tok=` do`  expl=`Thai/English code snippet about parsing string to array in Java, asking for method to remove all special characters from a string.

The phrase "I want to do som`

## Qualitative — 10 worst-reconstructed individual samples

- cos=0.368 pos=123 ‖v‖=112.0 tok=`7`  expl=`Structured Python documentation format with a code block showing a numeric API, following a Wikipedia-style definition pattern with UI element attributes.

The `
- cos=0.481 pos=491 ‖v‖=122.5 tok=`''`  expl=`<explanation>
Latin/Code error post with user trying to run a script for a discord bot, explaining issues with imports and not working properly.

The sentence s`
- cos=0.490 pos=118 ‖v‖=124.0 tok=`ab`  expl=`Unix documentation format with a quoted UI element showing a Rails code snippet, implying a search or browsing context about a specific application's UI appeara`
- cos=0.500 pos=505 ‖v‖=137.1 tok=` stub`  expl=`Greek/English language forum post with structured sections showing Java code for a DVD player, now presenting a second class "Product" with a constructor and su`
- cos=0.502 pos=366 ‖v‖=106.8 tok=`6`  expl=`Technical UI code snippet with chat format showing Python script context, using a UI component to display notifications and logs.

The example date range "比如今天的`
- cos=0.530 pos=319 ‖v‖=135.2 tok=`.`  expl=`Structured data table format with numbered statistics and formatted text blocks continuing a Python library report pattern, with a specific chart title and form`
- cos=0.551 pos=469 ‖v‖=118.6 tok=`={`  expl=`Structured Django/WordPress post with metadata and content blocks, now displaying a formatted list of social media links with a Twitter widget.

The quoted code`
- cos=0.552 pos=462 ‖v‖=114.9 tok=` |`  expl=`Technical UI documentation format with numbered list of Java compiler errors, showing formatted hexadecimal values with Unicode characters and their decoded rep`
- cos=0.559 pos=112 ‖v‖=105.8 tok=`3`  expl=`Structured reference format with numbered sections and quoted text from a JavaScript library, following a pattern of code blocks and formatted definitions.

The`
- cos=0.562 pos=156 ‖v‖=86.7 tok=` the`  expl=`Python code snippet with variable formatting and list comprehension context, informal tone with error messages and troubleshooting advice about `itertools` and `

## Qualitative — 10 random samples

- cos=0.883 pos=31 ‖v‖=131.7 tok=`/C`  expl=`Indian developer blog format with technical discussion about Android SDK tools, establishing context around "C++/C++" compiler output.

The phrase "So the quest`
- cos=0.877 pos=251 ‖v‖=132.1 tok=`Stop`  expl=`Technical .NET forum post with code examples demonstrating a WPF control, mixing UI elements and XAML markup.

The quoted XAML snippet mid-sentence ("Brush is s`
- cos=0.932 pos=45 ‖v‖=126.7 tok=` had`  expl=`Technical Linux UI post format with casual tone describing software behavior, establishing context around apt-get update/upgrade notifications.

The sentence "I`
- cos=0.888 pos=21 ‖v‖=128.0 tok=`Session`  expl=`.NET Core documentation format with code example pattern, describing `IPage<T>` behavior for ASP.NET MVC.

The phrase "A component that requires ISessionAware a`
- cos=0.923 pos=24 ‖v‖=115.3 tok=` a`  expl=`Discord.py bot code snippet with Python syntax, implying a user is trying to create a command for a discord bot using discord.py library.

The phrase "So i'm tr`
- cos=0.917 pos=112 ‖v‖=131.6 tok=` this`  expl=`Technical GIS/Python question about raster data extraction, showing a script with missing files and corrupted shapefiles, implying a data source or dataset cont`
- cos=0.862 pos=113 ‖v‖=126.3 tok=`.\n\n\n`  expl=`Technical/StackExchange answer format with a Windows UI warning about Chrome's memory usage, suggesting a user is explaining a specific behavior or workaround.
`
- cos=0.909 pos=273 ‖v‖=115.7 tok=` all`  expl=`French/PHP forum post about a CMS block display problem with "admin" class in a list view, explaining a jQuery issue with pagination.

The sentence structure "S`
- cos=0.817 pos=135 ‖v‖=83.9 tok=`6`  expl=`Technical SQL query format with code snippet showing a generator function to calculate sequence of multiples of a given number, using LINQ and SQL.

The phrase `
- cos=0.909 pos=27 ‖v‖=124.4 tok=` data`  expl=`Greek/JavaScript code snippet with tags suggesting a chart or form issue, implying a question about axios fetch and chartjs usage.

The phrase "i try to send ar`