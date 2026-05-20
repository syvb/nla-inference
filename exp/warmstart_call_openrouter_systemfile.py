"""Warm-start caller that takes a system prompt verbatim from a file.

No few-shot logic — examples are assumed to be hardcoded into the system
prompt by the file's author. Each call has the system prompt + a single
user message containing just the input text (wrapped in
<begin_text>...</end_text>).

System prompt is cached via Anthropic ephemeral cache_control. Two
sequential warm calls prime the cache before going concurrent.
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


def _extract_inner_text(full_prompt: str) -> str:
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
    system_prompt = Path(args.system_file).read_text()
    print(f"system prompt: {len(system_prompt)} chars (from {args.system_file})")

    table = pq.read_table(args.input)
    prompts = table.column("warmstart_prompt").to_pylist()
    sample_idx = table.column("sample_idx").to_pylist()
    n = len(prompts) if args.limit <= 0 else min(args.limit, len(prompts))
    prompts = prompts[:n]
    sample_idx = sample_idx[:n]

    user_texts = [_extract_inner_text(p) for p in prompts]
    print(f"calling {args.model} on {n} eval prompts (concurrency={args.concurrency})")

    sem = asyncio.Semaphore(args.concurrency)
    usage_log: list[dict] = []
    t0 = time.time()

    async with httpx.AsyncClient() as client:
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
    ap.add_argument("--system-file", required=True, help="file containing the system prompt verbatim")
    ap.add_argument("--output", required=True)
    ap.add_argument("--key-file", default="~/.openrouter_key")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
