"""Score the logprob of each sampled assistant token under google/gemma-4-26b-a4b-it.

For each sample where role=='asst_content':
  1. Look up the original conversation by conv_hash from regen_wildchat_20k.jsonl.
  2. Find which assistant turn contains the sampled token by searching for
     chars_before in the conversation text.
  3. Truncate that assistant turn's text at the token boundary.
  4. Build messages: [...prior msgs, last user msg, partial assistant msg].
  5. Call OpenRouter chat completion with logprobs=True, top_logprobs=20.
  6. Match token_str against the returned top-20 candidates.

Notes on tokenizer:
  - Gemma-3 and Gemma-4 share the content vocab (262144 IDs map identically
    for ordinary tokens). They differ on chat-control tokens, but we skip
    those (only score asst_content).
  - Matching is by exact string equality. We also report whether a candidate
    is a prefix of token_str (multi-token case) for visibility.

Output: a new parquet with the original columns + scoring columns.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-4-26b-a4b-it"
PROVIDER_PREFS = {"order": ["DekaLLM", "NextBit", "Venice"], "allow_fallbacks": False}


def load_conv_index(jsonl_path: Path) -> dict[str, list[dict]]:
    """Return {conv_hash: [messages]} for every conv in the regen file."""
    idx: dict[str, list[dict]] = {}
    with jsonl_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ch = rec.get("conv_hash")
            if ch:
                idx[ch] = rec.get("messages") or []
    return idx


def build_prefill_messages(messages: list[dict], chars_before: str,
                            token_str: str) -> tuple[list[dict], str] | None:
    """Find which message contains chars_before+token_str, return:
        (messages_for_api, partial_asst_text)
    where messages_for_api = [...prior messages, ...truncated current message OR None]
    and partial_asst_text = the assistant turn's content up to the token boundary.

    Returns None if the chars_before can't be located in any assistant message.
    """
    target = chars_before + token_str
    # Search from the LAST asst message backward — chars_before is the immediate
    # context of the token, so it's right next to it. We want the latest
    # occurrence in case the same chars_before appears earlier.
    for i in reversed(range(len(messages))):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        # Find LAST occurrence of chars_before in this message
        idx = content.rfind(chars_before)
        if idx < 0:
            continue
        prefix_end = idx + len(chars_before)
        # The next chars after chars_before should match token_str
        # (could differ slightly due to whitespace normalization in regen, but rare).
        # If exact match fails, accept anyway — chars_before is usually unique.
        partial_asst = content[:prefix_end]
        prior_msgs = list(messages[:i])
        return prior_msgs, partial_asst
    return None


async def score_one(client: httpx.AsyncClient, api_key: str, messages_prior: list[dict],
                     partial_asst: str, token_str: str, *,
                     sem: asyncio.Semaphore, max_retries: int = 4) -> dict:
    """One API call. Returns dict with scoring fields."""
    msgs = list(messages_prior) + [{"role": "assistant", "content": partial_asst}]
    body = {
        "model": MODEL,
        "messages": msgs,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
        "provider": PROVIDER_PREFS,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                    content=orjson.dumps(body),
                )
            if resp.status_code == 429:
                await asyncio.sleep(2.0 * (2 ** attempt))
                continue
            resp.raise_for_status()
            out = resp.json()
            choices = out.get("choices") or []
            if not choices:
                last_err = "no choices"
                await asyncio.sleep(1.0 * (2 ** attempt))
                continue
            lp_block = choices[0].get("logprobs")
            if not lp_block or not lp_block.get("content"):
                last_err = "no logprobs"
                await asyncio.sleep(1.0 * (2 ** attempt))
                continue
            top = lp_block["content"][0]["top_logprobs"]
            top1 = top[0]
            # Find exact match
            target_logprob = None
            target_rank = None
            match_type = "not_found"
            for r, tk in enumerate(top, start=1):
                if tk["token"] == token_str:
                    target_logprob = tk["logprob"]
                    target_rank = r
                    match_type = "exact"
                    break
            # If not exact, find best prefix relationship
            best_prefix_logprob = None
            best_prefix_token = None
            best_prefix_rank = None
            best_prefix_type = None  # "candidate_is_prefix" / "target_is_prefix"
            if match_type == "not_found":
                for r, tk in enumerate(top, start=1):
                    cand = tk["token"]
                    if not cand:
                        continue
                    if token_str.startswith(cand) and len(cand) < len(token_str):
                        # Candidate is a prefix of target — multi-token case
                        if best_prefix_logprob is None or tk["logprob"] > best_prefix_logprob:
                            best_prefix_logprob = tk["logprob"]
                            best_prefix_token = cand
                            best_prefix_rank = r
                            best_prefix_type = "candidate_is_prefix"
                    elif cand.startswith(token_str) and len(cand) > len(token_str):
                        # Target is prefix of candidate — single-token covers target+more
                        if best_prefix_logprob is None or tk["logprob"] > best_prefix_logprob:
                            best_prefix_logprob = tk["logprob"]
                            best_prefix_token = cand
                            best_prefix_rank = r
                            best_prefix_type = "target_is_prefix"
                if best_prefix_logprob is not None:
                    match_type = best_prefix_type
            return dict(
                top1_token=top1["token"],
                top1_logprob=top1["logprob"],
                target_logprob=target_logprob,
                target_rank=target_rank,
                in_top20=(target_rank is not None),
                match_type=match_type,
                best_prefix_token=best_prefix_token,
                best_prefix_logprob=best_prefix_logprob,
                best_prefix_rank=best_prefix_rank,
                top20_min_logprob=top[-1]["logprob"] if top else None,
                provider=out.get("provider"),
                error=None,
            )
        except (httpx.HTTPError, httpx.ReadError, httpx.RemoteProtocolError,
                httpx.ConnectError, httpx.PoolTimeout) as e:
            last_err = repr(e)[:120]
            await asyncio.sleep(0.5 * (2 ** attempt))
    return dict(
        top1_token=None, top1_logprob=None, target_logprob=None,
        target_rank=None, in_top20=False, match_type="error",
        best_prefix_token=None, best_prefix_logprob=None, best_prefix_rank=None,
        top20_min_logprob=None, provider=None, error=last_err,
    )


async def run(args):
    api_key = Path(os.path.expanduser("~/.openrouter_key")).read_text().strip()

    print(f"[score] loading conv index from {args.regen_jsonl}")
    conv_idx = load_conv_index(Path(args.regen_jsonl))
    print(f"[score] {len(conv_idx)} convs in index")

    print(f"[score] loading results parquet from {args.parquet}")
    tbl = pq.read_table(args.parquet, columns=[
        "sample_idx", "conv_hash", "role", "token_id", "token_str",
        "position", "unpadded_pos", "chars_before", "chars_after",
    ])
    df = tbl.to_pandas()
    n_total = len(df)
    print(f"[score] {n_total} rows")

    # Filter to asst_content
    asst = df[df.role == "asst_content"].reset_index(drop=True)
    print(f"[score] {len(asst)} asst_content rows to score")

    # Output schema: keep original cols + new scoring cols
    new_cols = {
        "top1_token": [None] * len(asst),
        "top1_logprob": [None] * len(asst),
        "target_logprob": [None] * len(asst),
        "target_rank": [None] * len(asst),
        "in_top20": [False] * len(asst),
        "match_type": [None] * len(asst),
        "best_prefix_token": [None] * len(asst),
        "best_prefix_logprob": [None] * len(asst),
        "best_prefix_rank": [None] * len(asst),
        "top20_min_logprob": [None] * len(asst),
        "provider": [None] * len(asst),
        "error": [None] * len(asst),
    }

    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency * 4,
                          max_keepalive_connections=args.concurrency)

    t0 = time.time()
    n_done = 0
    n_skip_noconv = 0
    n_skip_nomsg = 0

    async with httpx.AsyncClient(limits=limits,
                                  timeout=httpx.Timeout(args.timeout)) as client:
        async def process(row_idx: int):
            nonlocal n_done, n_skip_noconv, n_skip_nomsg
            r = asst.iloc[row_idx]
            conv = conv_idx.get(r.conv_hash)
            if not conv:
                n_skip_noconv += 1
                new_cols["error"][row_idx] = "no_conv"
                n_done += 1
                return
            built = build_prefill_messages(conv, r.chars_before, r.token_str)
            if built is None:
                n_skip_nomsg += 1
                new_cols["error"][row_idx] = "no_msg_match"
                n_done += 1
                return
            messages_prior, partial_asst = built
            result = await score_one(client, api_key, messages_prior, partial_asst,
                                      r.token_str, sem=sem)
            for k, v in result.items():
                new_cols[k][row_idx] = v
            n_done += 1
            if n_done % 200 == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-6)
                eta = (len(asst) - n_done) / max(rate, 1e-6)
                print(f"[score] {n_done}/{len(asst)}  "
                      f"skip_noconv={n_skip_noconv} skip_nomsg={n_skip_nomsg}  "
                      f"{rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

        tasks = [asyncio.create_task(process(i)) for i in range(len(asst))]
        await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    print(f"[score] done. {n_done} scored in {elapsed/60:.1f}min  "
          f"(noconv={n_skip_noconv}, nomsg={n_skip_nomsg})")

    # Build output table
    for k, vs in new_cols.items():
        asst[k] = vs

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(asst, preserve_index=False),
                   out_path, compression="zstd")
    print(f"[score] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Quick summary
    in_top = sum(1 for v in new_cols["in_top20"] if v)
    err = sum(1 for v in new_cols["error"] if v)
    target_lps = [v for v in new_cols["target_logprob"] if v is not None]
    print(f"[score] in_top20: {in_top} ({100*in_top/len(asst):.1f}%)  "
          f"errors: {err}  target_logprob_count: {len(target_lps)}")
    if target_lps:
        arr = np.array(target_lps)
        print(f"  target_logprob: mean={arr.mean():.3f} median={np.median(arr):.3f} "
              f"p10={np.percentile(arr,10):.3f} p90={np.percentile(arr,90):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--regen-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
