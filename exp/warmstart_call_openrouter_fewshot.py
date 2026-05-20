"""Few-shot warm-start caller with prompt caching.

Prepends 4 (user, assistant) turns from `--fewshot-json` to each call, with
cache_control on the last few-shot assistant turn (cumulative cache:
everything before & including is cached). Eval prompt is the final user
turn (varies per call).

Reports cache stats (creation/read tokens) from OpenRouter's usage block.
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


def build_messages(shots: list[dict], eval_prompt: str) -> list[dict]:
    """Build messages with cache_control on the LAST few-shot assistant turn."""
    msgs: list[dict] = []
    for i, s in enumerate(shots):
        msgs.append({"role": "user", "content": [{"type": "text", "text": s["prompt"]}]})
        asst_content: dict = {"type": "text", "text": s["response"]}
        if i == len(shots) - 1:
            asst_content["cache_control"] = {"type": "ephemeral"}
        msgs.append({"role": "assistant", "content": [asst_content]})
    msgs.append({"role": "user", "content": eval_prompt})
    return msgs


async def call_one(client, sem, key, model, messages, max_tokens, usage_log):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 1.0,
                        # OpenRouter passes anthropic-specific fields via this
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
    print(f"loaded {len(shots)} fewshot shots")
    for i, s in enumerate(shots):
        print(f"  shot {i}: sample_idx={s['sample_idx']}  prompt_chars={len(s['prompt'])}  "
              f"response_chars={len(s['response'])}")

    table = pq.read_table(args.input)
    prompts = table.column("warmstart_prompt").to_pylist()
    sample_idx = table.column("sample_idx").to_pylist()
    n = len(prompts) if args.limit <= 0 else min(args.limit, len(prompts))
    prompts = prompts[:n]
    sample_idx = sample_idx[:n]
    print(f"calling {args.model} on {n} eval prompts (concurrency={args.concurrency})")

    sem = asyncio.Semaphore(args.concurrency)
    usage_log: list[dict] = []
    t0 = time.time()

    async with httpx.AsyncClient() as client:
        # Warm the cache: make a single call FIRST so subsequent concurrent calls
        # all hit the same cache key. (Otherwise the first ~concurrency calls
        # would each independently write the cache.)
        print("  warming cache with 1 sync call...")
        warm_msgs = build_messages(shots, prompts[0])
        warm_result = await call_one(client, sem, key, args.model, warm_msgs, args.max_tokens, usage_log)
        print(f"  cache warmed (elapsed={time.time()-t0:.1f}s)")
        if usage_log:
            u = usage_log[0]
            print(f"  warm-call usage: {u}")

        results: list[str | None] = [warm_result] + [None] * (n - 1)
        tasks = [
            asyncio.create_task(
                call_one(client, sem, key, args.model,
                         build_messages(shots, p), args.max_tokens, usage_log)
            )
            for p in prompts[1:]
        ]
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if done % 25 == 0 or done == n - 1:
                print(f"  {done}/{n-1}  elapsed={time.time()-t0:.0f}s", flush=True)
        for i, t in enumerate(tasks):
            results[i + 1] = t.result()

    print(f"done in {time.time()-t0:.1f}s")

    # Cache stats — sum cache_read / cache_creation tokens across calls
    total_input = sum(u.get("prompt_tokens", 0) for u in usage_log)
    total_output = sum(u.get("completion_tokens", 0) for u in usage_log)
    # Anthropic-style: prompt_tokens_details.cached_tokens (OpenRouter passes through)
    cache_read = 0
    cache_creation = 0
    for u in usage_log:
        details = u.get("prompt_tokens_details", {}) or {}
        cache_read += details.get("cached_tokens", 0)
        cache_creation += u.get("cache_creation_input_tokens", 0)
    print()
    print(f"USAGE across {len(usage_log)} calls:")
    print(f"  prompt_tokens total:           {total_input:,}")
    print(f"  completion_tokens total:       {total_output:,}")
    print(f"  cache_read_tokens total:       {cache_read:,}")
    print(f"  cache_creation_tokens total:   {cache_creation:,}")
    if total_input:
        print(f"  cache hit rate (read/input):   {100*cache_read/total_input:.1f}%")

    cleaned = [extract_and_clean(r) if isinstance(r, str) and not r.startswith("__ERR_") else None for r in results]
    raw = [r if r is not None else "" for r in results]
    err_counts = Counter()
    for r in results:
        if r is None:
            err_counts["None"] += 1
        elif r.startswith("__ERR_"):
            err_counts[r.split(":")[0]] += 1
    n_ok = sum(1 for c in cleaned if c is not None)
    print(f"  parsed OK:        {n_ok}/{n}")
    if err_counts:
        print(f"  API errors:       {dict(err_counts)}")
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
    ap.add_argument("--limit", type=int, default=-1, help="cap eval prompts; -1 = all")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
