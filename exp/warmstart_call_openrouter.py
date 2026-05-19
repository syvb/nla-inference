"""Call OpenRouter (Sonnet 4.6) on warm-start prompts, clean per stage2 logic.

Concurrent via httpx + asyncio. Writes per-row results parquet with both raw
and cleaned explanations. Rows that fail extraction get cleaned=None and
are kept in the parquet for diagnostics — downstream filters them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

# Verbatim from stage2_api_explain.py
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


async def call_one(client, sem, key, model, prompt, max_tokens):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 1.0,
                    },
                    timeout=60.0,
                )
                if r.status_code == 200:
                    d = r.json()
                    return d["choices"][0]["message"]["content"]
                elif r.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(2 ** attempt)
                else:
                    return f"__ERR_{r.status_code}__:{r.text[:200]}"
            except (httpx.TimeoutException, httpx.HTTPError) as e:
                await asyncio.sleep(2 ** attempt)
        return None


async def amain(args):
    key = Path(os.path.expanduser(args.key_file)).read_text().strip()
    table = pq.read_table(args.input)
    prompts = table.column("warmstart_prompt").to_pylist()
    sample_idx = table.column("sample_idx").to_pylist()
    n = len(prompts)
    print(f"loaded {n} prompts; calling {args.model} with concurrency={args.concurrency}")

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    async with httpx.AsyncClient() as client:
        results = [None] * n
        tasks = [
            asyncio.create_task(call_one(client, sem, key, args.model, p, args.max_tokens))
            for p in prompts
        ]
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro  # just to update progress — we collect by index below
            done += 1
            if done % 25 == 0 or done == n:
                print(f"  {done}/{n}  elapsed={time.time()-t0:.0f}s", flush=True)
        for i, t in enumerate(tasks):
            results[i] = t.result()

    print(f"done in {time.time()-t0:.1f}s")

    cleaned = [extract_and_clean(r) if isinstance(r, str) and not r.startswith("__ERR_") else None for r in results]
    raw = [r if r is not None else "" for r in results]
    n_ok = sum(1 for c in cleaned if c is not None)
    n_err = sum(1 for r in results if r is None or (isinstance(r, str) and r.startswith("__ERR_")))
    n_dropped = n - n_ok - n_err
    print(f"  parsed OK:        {n_ok}/{n}")
    print(f"  API errors/nulls: {n_err}/{n}")
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
    ap.add_argument("--input", required=True, help="prompts_500.parquet")
    ap.add_argument("--output", required=True)
    ap.add_argument("--key-file", default="~/.openrouter_key")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
