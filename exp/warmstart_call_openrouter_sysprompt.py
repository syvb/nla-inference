"""System-prompt few-shot warm-start caller.

All instructions (task description, format spec, examples) live in the system
prompt. The user message is ONLY the text to analyze (wrapped in
<begin_text>...</end_text> markers). The model is told to match the example
outputs as closely as possible.

System prompt structure:
  [task description + format spec]
  [5 example input/output pairs]
  [explicit instruction to match the example output style]

The system prompt is cached (it's constant across all eval calls).

We keep the <analysis>...</analysis> tag convention so the parser matches.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

_DEFAULT_RESPONSE_PATTERN = r"<analysis>\s*(.*?)\s*</analysis>"
_MIN_FEATURES = 2
_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"[-*•+–—]"
    r"|\d+[.)]"
    r"|\(\d+\)"
    r"|[a-zA-Z][.)]"
    r"|\([a-zA-Z]\)"
    r"|[ivxIVX]+[.)]"
    r")\s+"
)
_BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*\s*")


def extract_and_clean(raw: str, pattern: str = _DEFAULT_RESPONSE_PATTERN) -> str | None:
    m = re.search(pattern, raw, flags=re.DOTALL)
    if m is None:
        return None
    content = m.group(1)
    cleaned: list[str] = []
    for line in content.split("\n"):
        line = _LIST_PREFIX_RE.sub("", line)
        line = _BOLD_WRAP_RE.sub(r"\1 ", line)
        line = line.strip().strip("*_")
        if line:
            cleaned.append(line)
    out = "\n\n".join(cleaned)
    if out.count("\n\n") + 1 < _MIN_FEATURES:
        return None
    return out


# Same task/format text as stage2's _DEFAULT_INSTRUCTION, minus the
# "Text to analyze:" footer (which is now the user's message).
_TASK_INSTRUCTIONS = """You will be shown a text snippet, and asked to predict the most important features a language model would use to predict what text comes next.

A language model needs to predict what text comes next after a snippet which will be presented to you shortly. Identify the 2-3 most important features it would use for this prediction.
Focus on what the language model must be "thinking about" at the point where the provided text ends. You should not need to reference the fact that the text is truncated/incomplete/a prefix: the language model is causal, so only sees the prefix to what it predicts and this is implicit.
Order features by what is most important for predicting the next tokens. Each feature should consist of a concise ~10-20 word description. Feel free to include specific textual examples inline.

Feature types to consider (as inspiration, not a rigid checklist):
- Syntactic/structural constraints: "unclosed parenthesis requires matching close"
- Immediate semantic expectations: "list promised three items but only two given"
- Stylistic/register patterns: "formal academic tone maintained throughout"
- Narrative/argumentative momentum: "thesis stated, supporting evidence now expected"
- Domain/genre signals: "medical case history following SOAP format"
- Repetition/continuation patterns: "same phrase structure repeating with variations"

The final feature must describe the very end of the presented sequence: its role, what it's part of, and immediate constraints on what follows.

Format — IMPORTANT: keep to ~80-100 words total and ALWAYS close the tag:
<analysis>
[first feature — include specific examples when relevant]
[second feature]
[final feature: the last token, its role, immediate constraints]
</analysis>"""


def build_system_prompt(shots: list[dict]) -> str:
    """Construct the system prompt: task spec + N example pairs + final reminder."""
    parts = [_TASK_INSTRUCTIONS, ""]
    parts.append(f"Below are {len(shots)} example input/output pairs. Your response "
                 f"should match the style, length, structure, and tone of the example "
                 f"outputs as closely as possible — same kind of features, same level "
                 f"of detail, same prose register.")
    parts.append("")
    for i, s in enumerate(shots, 1):
        parts.append(f"--- Example {i} ---")
        parts.append("Input:")
        parts.append(f"<begin_text>{_extract_inner_text(s['prompt'])}<end_text>")
        parts.append("")
        parts.append("Output:")
        parts.append(s["response"])
        parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("Now produce an <analysis>...</analysis> response for the user's text "
                 "below. Match the style of the examples above as closely as possible.")
    return "\n".join(parts)


def _extract_inner_text(full_prompt: str) -> str:
    """Pull just the <begin_text>...<end_text> contents from a stage2-format prompt."""
    m = re.search(r"<begin_text>(.*?)<end_text>", full_prompt, flags=re.DOTALL)
    if m is None:
        raise ValueError(f"could not find <begin_text>...<end_text> in prompt:\n{full_prompt[-300:]!r}")
    return m.group(1)


async def call_one(client, sem, key, model, system, user_text, max_tokens, usage_log):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": [
                                {"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}},
                            ]},
                            {"role": "user", "content": f"<begin_text>{user_text}<end_text>"},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 1.0,
                        "usage": {"include": True},
                    },
                    timeout=120.0,
                )
                if r.status_code == 200:
                    d = r.json()
                    if "usage" in d:
                        usage_log.append(d["usage"])
                    return d["choices"][0]["message"]["content"]
                elif r.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(2 ** attempt)
                else:
                    return f"__ERR_{r.status_code}__:{r.text[:300]}"
            except (httpx.TimeoutException, httpx.HTTPError):
                await asyncio.sleep(2 ** attempt)
        return None


async def amain(args):
    key = Path(os.path.expanduser(args.key_file)).read_text().strip()
    shots = json.loads(Path(args.fewshot_json).read_text())
    if args.n_shots > 0:
        shots = shots[:args.n_shots]
    print(f"using {len(shots)} fewshot shots from {args.fewshot_json}")

    system_prompt = build_system_prompt(shots)
    print(f"system prompt size: {len(system_prompt)} chars")
    print(f"--- system prompt head ({min(800, len(system_prompt))} chars) ---")
    print(system_prompt[:800])
    print("...")
    print(f"--- system prompt tail (300 chars) ---")
    print(system_prompt[-300:])
    print()

    table = pq.read_table(args.input)
    prompts = table.column("warmstart_prompt").to_pylist()
    sample_idx = table.column("sample_idx").to_pylist()
    n = len(prompts) if args.limit <= 0 else min(args.limit, len(prompts))
    prompts = prompts[:n]
    sample_idx = sample_idx[:n]

    # Pre-extract user texts from prompts
    user_texts = [_extract_inner_text(p) for p in prompts]
    print(f"calling {args.model} on {n} eval prompts (concurrency={args.concurrency})")

    sem = asyncio.Semaphore(args.concurrency)
    usage_log: list[dict] = []
    t0 = time.time()

    async with httpx.AsyncClient() as client:
        # Warm the cache with TWO sequential calls — Anthropic's cache write
        # appears to take a moment to become queryable after the response
        # returns, so a single warm call followed by a concurrent burst still
        # has most of the burst miss the cache.
        print("  warming cache (2 sequential calls)...")
        warm0 = await call_one(client, sem, key, args.model, system_prompt,
                               user_texts[0], args.max_tokens, usage_log)
        warm1 = await call_one(client, sem, key, args.model, system_prompt,
                               user_texts[1] if n > 1 else user_texts[0],
                               args.max_tokens, usage_log)
        print(f"  cache warmed in {time.time()-t0:.1f}s")
        for u in usage_log[:2]:
            d = u.get("prompt_tokens_details", {}) or {}
            print(f"    warm: cached={d.get('cached_tokens',0)} "
                  f"write={d.get('cache_write_tokens',0)} input={u.get('prompt_tokens',0)}")

        results: list[str | None] = [warm0]
        start_idx = 1
        if n > 1:
            results.append(warm1)
            start_idx = 2
        results.extend([None] * (n - start_idx))
        tasks = [
            asyncio.create_task(call_one(client, sem, key, args.model, system_prompt,
                                         user_texts[i], args.max_tokens, usage_log))
            for i in range(start_idx, n)
        ]
        n_concurrent = len(tasks)
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if done % 25 == 0 or done == n_concurrent:
                print(f"  {done}/{n_concurrent}  elapsed={time.time()-t0:.0f}s", flush=True)
        for i, t in enumerate(tasks):
            results[i + start_idx] = t.result()

    print(f"done in {time.time()-t0:.1f}s")

    total_input = sum(u.get("prompt_tokens", 0) for u in usage_log)
    total_output = sum(u.get("completion_tokens", 0) for u in usage_log)
    cache_read = 0
    for u in usage_log:
        details = u.get("prompt_tokens_details", {}) or {}
        cache_read += details.get("cached_tokens", 0)
    print()
    print(f"USAGE across {len(usage_log)} calls:")
    print(f"  prompt_tokens total:           {total_input:,}")
    print(f"  completion_tokens total:       {total_output:,}")
    print(f"  cache_read_tokens total:       {cache_read:,}")
    if total_input:
        print(f"  cache hit rate (read/input):   {100*cache_read/total_input:.1f}%")

    cleaned = [extract_and_clean(r) if isinstance(r, str) and not r.startswith("__ERR_") else None for r in results]
    raw = [r if r is not None else "" for r in results]
    err_counts = Counter()
    for r in results:
        if r is None: err_counts["None"] += 1
        elif r.startswith("__ERR_"): err_counts[r.split(":")[0]] += 1
    n_ok = sum(1 for c in cleaned if c is not None)
    print(f"  parsed OK:        {n_ok}/{n}")
    if err_counts: print(f"  API errors:       {dict(err_counts)}")
    n_dropped = n - n_ok - sum(err_counts.values())
    print(f"  extract failures: {n_dropped}/{n}")

    out = pa.table({
        "sample_idx": pa.array(sample_idx, type=pa.int64()),
        "warmstart_raw": pa.array(raw),
        "warmstart_cleaned": pa.array([c if c is not None else "" for c in cleaned]),
        "warmstart_parsed": pa.array([c is not None for c in cleaned], type=pa.bool_()),
    })
    pq.write_table(out, args.output, compression="zstd")
    print(f"wrote {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="eval prompts parquet")
    ap.add_argument("--fewshot-json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--key-file", default="~/.openrouter_key")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--n-shots", type=int, default=5, help="how many shots to use from the json (0=all)")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
