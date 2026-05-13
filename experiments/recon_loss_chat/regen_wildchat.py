"""Sample WildChat-1M and regenerate assistant turns with Gemma-3-12B via OpenRouter.

For each sampled conversation:
  1. Take the WildChat user turns in order.
  2. Sequential replay: regenerate each assistant turn given (u1, a1_gemma, u2, ..., uN)
     so the conversation history seen by Gemma is internally consistent with
     its own prior outputs (not WildChat's original assistant text).
  3. Save: {conv_hash, lang, n_turns, messages: [{role, content}, ...]}.

Filter: language == 'English' at the top level. No toxicity / redaction filter.

Reads OpenRouter API key from ~/.openrouter_key. Uses async + a Semaphore for
concurrency. Sequential regen WITHIN a conversation, but many conversations
run in parallel.

Output: experiments/recon_loss_chat/regen_wildchat_<N>.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx
import orjson


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-3-12b-it"


def stream_english(seed: int, n_target: int, max_skip: int = 200_000):
    """Yield English WildChat conversations from streaming load, reservoir-sampled.

    Simple approach: stream the dataset, pre-filter to English, and yield the
    first n_target that pass. WildChat-1M is shuffled enough at source that this
    gives a reasonable sample without needing a full reservoir.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    # Even-larger shuffle buffer here is fine — we hold tokens-of-text-only metadata
    ds = ds.shuffle(seed=seed, buffer_size=20_000)

    seen = 0
    yielded = 0
    for ex in ds:
        seen += 1
        if seen > max_skip and yielded < n_target:
            # Should never hit this in practice
            print(f"[stream] hit max_skip={max_skip}, only {yielded} yielded so far",
                  file=sys.stderr, flush=True)
            return
        lang = ex.get("language") or ""
        if lang.lower() != "english":
            continue
        conv = ex.get("conversation") or []
        # Need at least 1 user + 1 assistant turn
        roles = [m.get("role") for m in conv]
        if "user" not in roles or "assistant" not in roles:
            continue
        yield ex
        yielded += 1
        if yielded >= n_target:
            return


async def regen_one(client: httpx.AsyncClient, conv: list[dict], api_key: str,
                    *, max_tokens: int, sem: asyncio.Semaphore,
                    max_retries: int = 4) -> list[dict] | None:
    """Sequential-replay regenerate every assistant turn in `conv` (in order).

    Input: full conversation = [user, assistant, user, assistant, ...]
    Returns: same shape but with assistant content replaced by Gemma's output
             given the (regenerated) prior turns.
    Raises only on hard failures (4 retries exhausted).
    """
    out_messages: list[dict] = []
    history_for_call: list[dict] = []

    for msg in conv:
        role = msg.get("role")
        if role == "user":
            text = msg.get("content") or ""
            out_messages.append({"role": "user", "content": text})
            history_for_call.append({"role": "user", "content": text})
        elif role == "assistant":
            # Call OpenRouter with current history, get a regenerated assistant turn.
            body = {
                "model": MODEL,
                "messages": history_for_call,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }
            last_err = None
            for attempt in range(max_retries):
                try:
                    async with sem:
                        resp = await client.post(
                            OPENROUTER_URL,
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            content=orjson.dumps(body),
                        )
                    if resp.status_code == 429:
                        # rate limited; back off
                        await asyncio.sleep(2.0 * (2 ** attempt))
                        continue
                    resp.raise_for_status()
                    out = resp.json()
                    if "choices" not in out or not out["choices"]:
                        last_err = f"empty choices: {out}"
                        await asyncio.sleep(1.0 * (2 ** attempt))
                        continue
                    text = (out["choices"][0]["message"].get("content") or "").strip()
                    if not text:
                        last_err = "empty text"
                        await asyncio.sleep(1.0 * (2 ** attempt))
                        continue
                    out_messages.append({"role": "assistant", "content": text})
                    history_for_call.append({"role": "assistant", "content": text})
                    break
                except (httpx.HTTPError, httpx.ReadError, httpx.RemoteProtocolError,
                        httpx.ConnectError, httpx.PoolTimeout) as e:
                    last_err = repr(e)
                    await asyncio.sleep(0.5 * (2 ** attempt))
            else:
                # All retries exhausted
                return None
        else:
            # 'system' or other rare roles — drop, WildChat doesn't have system
            continue

    return out_messages


async def run(args):
    api_key = Path(os.path.expanduser("~/.openrouter_key")).read_text().strip()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[regen] sampling {args.n} English convs from WildChat-1M  seed={args.seed}")
    print(f"[regen] model = {MODEL}  max_tokens = {args.max_tokens}  "
          f"concurrency = {args.concurrency}")
    print(f"[regen] output = {out_path}")

    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency * 4,
                          max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(args.timeout)

    n_done = 0
    n_fail = 0
    t0 = time.time()
    write_lock = asyncio.Lock()

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        with out_path.open("w") as fout:
            async def process(ex):
                nonlocal n_done, n_fail
                conv = ex.get("conversation") or []
                result = await regen_one(client, conv, api_key,
                                          max_tokens=args.max_tokens, sem=sem)
                async with write_lock:
                    if result is None:
                        n_fail += 1
                    else:
                        rec = {
                            "conv_hash": ex.get("conversation_hash"),
                            "language": ex.get("language"),
                            "n_user_turns": sum(1 for m in result if m["role"] == "user"),
                            "n_asst_turns": sum(1 for m in result if m["role"] == "assistant"),
                            "messages": result,
                        }
                        fout.write(orjson.dumps(rec).decode() + "\n")
                        n_done += 1
                        if n_done % 100 == 0:
                            elapsed = time.time() - t0
                            rate = n_done / max(elapsed, 1e-6)
                            eta = (args.n - n_done) / max(rate, 1e-6)
                            print(f"[regen] {n_done}/{args.n}  "
                                  f"failed={n_fail}  rate={rate:.2f}/s  "
                                  f"eta={eta/60:.1f}min", flush=True)

            # Fire off tasks as conversations arrive from the stream
            tasks = []
            for ex in stream_english(args.seed, args.n):
                tasks.append(asyncio.create_task(process(ex)))
                # Cap in-flight tasks
                if len(tasks) >= args.concurrency * 8:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = [t for t in tasks if not t.done()]
            await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    print(f"[regen] done. wrote {n_done} convs in {elapsed/60:.1f}min "
          f"(failed: {n_fail}) → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
