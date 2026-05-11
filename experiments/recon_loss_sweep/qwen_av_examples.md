# How does Qwen's AV verbalize tokens with no standalone meaning?

Concrete examples drawn from the 20k Qwen sweep, focused on the
**6,669 / 20,000 (33 %)** tokens classified as "meaningless" — pure digits,
single punctuation marks, single letters, whitespace runs, and code-syntax
fragments. These tokens have a mean cos of **0.854** (vs full-dataset 0.876),
so they're modestly worse on average and are over-represented in the
catastrophic-failure tail.

## The structural pattern AV always uses

In every example below — failure or success — the AV's explanation is built
from the same three parts:

1. **A genre / format claim** ("Technical Python documentation format with
   …", "Math forum post with LaTeX equations …").
2. **A made-up quoted phrase** allegedly seen in the document, used to
   anchor the explanation in something concrete-sounding.
3. **A "Final token X" prediction** — what specific next-token the AV
   thinks should follow.

The third part is the giveaway: **the AV is essentially trying to predict
what comes after the activation's token.** The token-id signal is weak for
short / structural tokens, so the prediction depends almost entirely on
whether the AV correctly identified the broader genre.

If genre is correct → prediction is reasonable → AR maps the explanation
back close to the original direction → high cos.

If genre is wrong → prediction is detailed but wrong-domain → AR maps to a
wrong direction → low cos.

## Catastrophic failures (cos < 0.55)

### `'7'` at position 123 / 512 — cos = 0.368

**Real context:** Java / Jsoup library, HTTP error.
```
"Jsoup crawler and HTTP error fetching URL
I am writing a crawler with Jsoup and this is the HTTP error I get:
org.jsoup.HttpStatusException: HTTP erro…"
```

**AV's explanation:** [confidently asserts Python documentation, totally
wrong domain]

> Structured Python documentation format with a code block showing a
> numeric API, following a Wikipedia-style definition pattern with UI
> element attributes. The phrase "java.util.concurrent.FutureTask\<T\>
> class, with method 'waitTime' showing options to check result after
> request" is a formatted table with a "7 options" label, completing a
> common pattern of "returnOptions" showing the timing behavior of a
> request. **Final token "7" ends an incomplete parenthetical reference
> ("7, (7"), part of the "option 7: results, with return value (7"
> clause structure, strongly expecting "15" or "20)" to close the
> numeric condition qualifier.**

**What's happening:** the actual `7` in the document is most likely the
beginning of an HTTP error code (e.g., `7`12, `7`40) — *not* "option 7".
The AV's prediction (expecting "15" or "20)") is wrong because it picked
the wrong genre.

### `"''"` at position 491 / 512 — cos = 0.481

**Real context:** Stack Overflow question about screenshots and quoted key
names like `''Space''`.

**AV's explanation:** [degenerate output — runs away into a 200-token wall
of repeated apostrophes]

> Latin/Code error post with user trying to run a script for a discord
> bot, explaining issues with imports and not working properly. The
> sentence structure "So i have tryed 'pip install' " suggests a second
> point or context about the user's installation … Final token
> ` "''''''''''''''''''''''''''''''''''''''''''''''''''''''''''…`
> [cuts off after 200 tokens of `'`s]

**What's happening:** runaway generation — the AV got into a token-repeat
loop. This is a generation pathology, not just topic confusion.

### `'.'` at position 319 / 512 — cos = 0.530

**Real context:** Ruby / Project Euler timing question.

**AV's explanation:** [Python data-table report, totally wrong]

> Structured data table format with numbered statistics and formatted
> text blocks continuing a Python library report pattern, with a
> specific chart title and formula. The quoted statistic mid-sentence
> ("Sum of Month 11 and 12: 0.3, with the average") follows a common
> math library format listing "Delta" values … **Final token "0." is
> mid-number in a parenthetical value ("0. ... plus 0. "), part of a
> third statistic item listing ("The sum of Trees, a new month's
> data"), immediately expecting "12" or "3D" or "area to sum".**

The actual context never mentions data tables, statistics, charts, or
Python — it's a Ruby timing post. The AV invented a fully detailed
alternate document.

## Poor reconstruction (cos 0.65–0.75)

### `'^'` at position 111 / 242 — cos = 0.714

**Real context:** Math overflow, matrix decomposition with `AB=BA`.

**AV's explanation:** [partially correct: math, matrix algebra, LaTeX]

> Old math forum post with LaTeX formatting showing matrix equations
> for differential operators, establishing a pattern of defining
> properties of symmetric matrices. The sentence structure "Thus we
> have: $A^2=I^2=A^T$ and $B^2=B^T$ so that $A^2+B^2$ is symmetric and
> $B^2$ is diagonal" mirrors a known matrix identity … **Final token
> "^" is mid-expression in the second repeated matrix term "$B^2B^$,"
> strongly expecting "2" to close the second diagonal matrix
> expression, continuing the pattern "B^2."**

The genre identification is right (math forum, LaTeX, matrices). The
detailed prediction ("$B^2B^$" expecting "2") is a guess that's plausible
but probably not what the actual `^` is doing. Partial credit → cos 0.71.

### `'0'` at position 404 / 512 — cos = 0.746

**Real context:** OpenCV Kalman filter Python code.

**AV's explanation:**

> Python code snippet with a function definition using a neural network
> class, showing a UI widget for a Kalman filter with parameter values.
> The pattern "x,y = [1,0,0,0] # xdot,ydot" mirrors a list of
> trigonometric equations from the original formula … **Final token "0"
> is mid-expression within a repeated parenthetical pair
> ("0,0,0,0...0"), directly continuing the second pair of sine terms
> ("ydot=0,0"), strongly expecting "1]" to close the second axis
> condition, completing the symmetric rotation pattern.**

Genre is right (Python, Kalman). The specific prediction (`"1]"`) might
or might not match — either way, the local-context prediction is approximate
rather than exact, hence cos in the 0.7s rather than the 0.9s.

## Typical (cos 0.85–0.90)

### `' v'` at position 298 / 512 — cos = 0.858

**Real context:** Go module Q&A with `go v1.19`.

**AV's explanation:** [correctly identifies Go ecosystem, dependency
versions]

> Technical CLI output format with structured logs showing Go package
> versioning and security metrics for a Kubernetes operator, listing
> dependencies with version numbers. The pattern "github.com/elastic/go-logr
> v" is mid-sentence listing a specific Go package version … **Final
> token " v" is a version identifier mid-quote ("github.com/Shopify/sarama
> v"), part of a dependency list item with a specific package name and
> version constraint clause ("go-retryablehttp v"), expecting
> continuation like "1.14.0" or "0.0.0+".**

Domain correct (Go modules, version pinning), specific guess is
plausible (`v1.14.0` is a typical pattern). Cos ~0.86.

### `'       '` (7 spaces) at position 245 / 512 — cos = 0.876

**Real context:** Django REST Framework / image upload via Postman.

**AV's explanation:** [DRF / Telegram bot — half-right, half-confabulation]

> Python code snippet with Django API context, showing a class method
> to fetch form data for a Telegram bot's POST request. The pattern
> "def get_data(self): data = { 'id': self.user.id, 'name':
> self.user.name," establishes a dictionary structure listing fields
> from the POST request … **Final token " " is mid-list-item in the
> original dictionary definition ("fields = { 'id': request.POST.get('id'),
> "), immediately expecting " 'message': request.POST.get('message') "
> or " 'other_fields': value" to complete the field list.**

For pure-whitespace tokens the AV is essentially guessing what kind of
Python construct the indentation belongs to. Domain correct, specific
prediction is generic-but-plausible.

## Good reconstruction (cos > 0.93)

### `'0'` at position 12 / 512 — cos = 0.934

**Real context:** Python loop summation with `s = 0`.

**AV's explanation:**

> Greek/Python code format with a prompt structure suggesting a math
> problem or loop, likely a script or code snippet involving summation
> or iteration. The line "s = 0" begins a variable declaration ("s =
> 0"), strongly implying a Python code block or variable assignment
> context, possibly a summation or loop initialization related to the
> given inequality or sequence. **Final token "0" ends an incomplete
> variable assignment ("s = 0"), part of a variable declaration or
> initial value setup line, immediately expecting continuation like
> " ; for i in range..." or " for n in numbers:" or "sum = 0" or "The
> variable represents the sum."**

This is the AV at its best for a meaningless token: it directly quotes
the actual line `s = 0` (because at position 12 the local context is
visible to the activation), correctly identifies the construct as a
loop initialiser, and predicts a `for`-loop continuation. The actual
next text is `M = 3 / for i in range(M): / s += 1/2*k**2` — the AV's
prediction matches the *type* of continuation almost exactly. Cos = 0.93.

### `'.'` at position 24 / 249 — cos = 0.942

**Real context:** SockJS version mismatch (`SockJS 1.0.3` vs `1.0.0`).

**AV's explanation:**

> Technical/JavaScript documentation format with a quoted title
> suggesting a library or plugin, implying a code snippet or library
> reference about Socket.IO. The phrase "I'm using socket.io 1.0."
> suggests a version number or specific library attribute is being
> cited … **Final token "1.0." is mid-number in a version or numeric
> value ("1.0."), part of a quoted attribute listing ("A jQuery plugin:
> 1.0."), strongly expecting continuation like "5" or "9" or "11" or
> "some other version."**

Genre identification correct (JavaScript, library versions, semver).
Predicted continuation ("5" / "9" / "11" / "some other version")
matches the actual `.3` and `.0` digits in the post. Cos = 0.94.

### `' a'` at position 81 / 512 — cos = 0.937

**Real context:** Unity 3D model as a button.

**AV's explanation:**

> … Final token "a" is an indefinite article mid-sentence ("Is there
> a"), part of a question structure ("Is there a way..."), strongly
> expecting a noun like "way to detect" or "method to assign" or
> "specific script component" or "way to make it transparent without
> mesh."

The Unity post does contain the phrase "Is there a way…" and the AV's
prediction matches the actual continuation. The leading-space `' a'`
token has more specific context (it's an indefinite article in a
specific sentence pattern) than a bare digit, so the AV grounds its
prediction better. Cos = 0.94.

## Summary

For meaningless tokens, **AV's explanation is a "what-comes-next"
prediction in disguise**, usually phrased as:

1. "Document is a [genre] about [topic]"
2. "An example phrase: [confabulated quote]"
3. "Final token X opens [some construct], expecting [Y / Z / W]"

The reconstruction quality is determined almost entirely by part 1. If the
AV correctly identifies the genre, parts 2–3 land in the right region of
representation space and the AR can reconstruct. If part 1 is wrong, parts
2–3 are detailed-but-wrong, and reconstruction fails proportionally.

**Why short / late position fails more:** at short tokens the activation
carries little token-identity signal, and at late positions the activation
is a compressed summary of hundreds of tokens. Together the AV has thin
evidence for genre, falls back on a guess, and confabulates a confident
description of the wrong domain. There is no fallback template — every
guess is fresh prose — but the underlying behavior (commit-confidently-
to-a-guess-when-uncertain) produces the same kind of failure.

This contrasts with Gemma, where the AV had memorised a canonical "I don't
know" template ("research paper review…") and emitted it verbatim when
uncertain. Qwen never gives the same wrong answer twice; Gemma always gave
the same wrong answer.
