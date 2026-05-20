"""Build v4-style warm-start prompts for PT data + the 4-shot file.

PT activations parquet has no full text or doc_id — only a 200-char
text_preview. We download the 31 selected shards from common-pile and
match docs by their first 200 chars to recover the full text, then
reconstruct the prefix the same way stage0 does (decode first pos+1 tokens
of the tokenized doc, skip_special_tokens=True).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# Same as v2 (no token hint).
_DEFAULT_INSTRUCTION = """A language model needs to predict what text comes next after a snippet which will be presented to you shortly. Identify the 2-3 most important features it would use for this prediction.
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
</analysis>

Text to analyze:

<begin_text>{text}<end_text>"""

SHARDS = [
    "arxiv_abstracts/arxiv_abstracts.chunk.29.jsonl.gz",
    "arxiv_papers/arxiv_papers.chunk.49.jsonl.gz",
    "biodiversity_heritage_library/biodiversity_heritage_library.chunk.33.jsonl.gz",
    "caselaw_access_project/caselaw_access_project.chunk.40.jsonl.gz",
    "cccc/cccc.chunk.07.jsonl.gz",
    "data_provenance_initiative/data_provenance_initiative.chunk.43.jsonl.gz",
    "doab/doab.chunk.32.jsonl.gz",
    "foodista/foodista.chunk.32.jsonl.gz",
    "github_archive/github_archive.chunk.45.jsonl.gz",
    "library_of_congress/library_of_congress.chunk.13.jsonl.gz",
    "libretexts/libretexts.chunk.17.jsonl.gz",
    "news/news.chunk.48.jsonl.gz",
    "oercommons/oercommons.chunk.02.jsonl.gz",
    "peS2o/peS2o.chunk.15.jsonl.gz",
    "pre_1929_books/pre_1929_books.chunk.05.jsonl.gz",
    "pressbooks/pressbooks.chunk.59.jsonl.gz",
    "project_gutenberg/project_gutenberg.chunk.45.jsonl.gz",
    "public_domain_review/public_domain_review.chunk.50.jsonl.gz",
    "pubmed/pubmed.chunk.55.jsonl.gz",
    "python_enhancement_proposals/python_enhancement_proposals.chunk.06.jsonl.gz",
    "regulations/regulations.chunk.01.jsonl.gz",
    "stackexchange/stackexchange.chunk.42.jsonl.gz",
    "stackv2_edu/stackv2_edu.chunk.32.jsonl.gz",
    "stackv2_html/stackv2_html.chunk.47.jsonl.gz",
    "ubuntu_irc/ubuntu_irc.chunk.57.jsonl.gz",
    "uk_hansard/uk_hansard.chunk.30.jsonl.gz",
    "usgpo/usgpo.chunk.13.jsonl.gz",
    "uspto/uspto.chunk.37.jsonl.gz",
    "wikimedia/wikimedia.chunk.44.jsonl.gz",
    "wikiteam/wikiteam.chunk.39.jsonl.gz",
    "youtube/youtube.chunk.14.jsonl.gz",
]


def download_shard(shard: str) -> str:
    return hf_hub_download(
        "common-pile/comma_v0.1_training_dataset",
        shard,
        repo_type="dataset",
    )


def find_text_field(d: dict) -> str | None:
    for k in ("text", "content", "body"):
        if k in d and isinstance(d[k], str):
            return d[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True, help="PT activations parquet")
    ap.add_argument("--results", required=True, help="PT results parquet (for AV explanation)")
    ap.add_argument("--tok-dir", required=True)
    ap.add_argument("--out-eval", required=True, help="eval prompts parquet")
    ap.add_argument("--out-fewshot", required=True, help="fewshot JSON")
    ap.add_argument("--limit-eval", type=int, default=500)
    ap.add_argument("--n-shots", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--fewshot-seed", type=int, default=42)
    args = ap.parse_args()

    print("[load] tokenizer")
    tok = AutoTokenizer.from_pretrained(args.tok_dir, trust_remote_code=True)

    print("[load] activations + results")
    acts = pq.read_table(args.activations, columns=[
        "sample_idx", "token_str", "position", "seq_len", "text_preview",
    ])
    res = pq.read_table(args.results, columns=["sample_idx", "explanation", "av_parsed"])
    n_total = acts.num_rows
    print(f"  {n_total} activation rows")

    # Pick eval + fewshot (disjoint sets, fewshot from rows where AV parsed)
    res_idx = np.asarray(res.column("sample_idx").to_pylist())
    res_map = {int(s): i for i, s in enumerate(res_idx)}

    rng_eval = np.random.default_rng(args.eval_seed)
    eval_pick = rng_eval.choice(n_total, size=min(args.limit_eval, n_total), replace=False)
    eval_pick = np.sort(eval_pick)
    eval_sids = set(int(acts.column("sample_idx")[int(i)].as_py()) for i in eval_pick)

    rng_fs = np.random.default_rng(args.fewshot_seed)
    fs_candidates = []
    for i in range(n_total):
        sid = int(acts.column("sample_idx")[i].as_py())
        if sid in eval_sids: continue
        if sid not in res_map: continue
        if not res.column("av_parsed")[res_map[sid]].as_py(): continue
        fs_candidates.append(i)
    # Oversample so we have buffer for shots that fail (pos > re-tokenized len).
    fs_pick = rng_fs.choice(fs_candidates, size=min(args.n_shots * 4, len(fs_candidates)), replace=False)
    print(f"  eval rows: {len(eval_pick)}; fewshot candidates (oversampled): {len(fs_pick)} (want {args.n_shots})")

    # Build a "needed text_preview → list of (idx, role)" lookup. role in
    # {"eval", "fewshot"}. We strip leading spaces because some shards add them
    # and we want to match the actual preview.
    needed: dict[str, list[tuple[int, str]]] = {}
    for i in eval_pick:
        i = int(i)
        prev = acts.column("text_preview")[i].as_py()
        needed.setdefault(prev, []).append((i, "eval"))
    for i in fs_pick:
        i = int(i)
        prev = acts.column("text_preview")[i].as_py()
        needed.setdefault(prev, []).append((i, "fewshot"))
    print(f"  unique previews needed: {len(needed)}")

    # Download shards in parallel
    print("[download] 31 common-pile shards (parallel)...")
    t0 = time.time()
    paths: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(download_shard, s): s for s in SHARDS}
        for fut in as_completed(futs):
            s = futs[fut]
            paths[s] = fut.result()
            print(f"  ok {s} ({(time.time()-t0):.0f}s)", flush=True)
    print(f"download done in {time.time()-t0:.0f}s")

    # Stream shards, build text_preview -> full_text map
    print("[scan] streaming shards to find needed docs...")
    found: dict[int, str] = {}  # acts row idx -> full text
    n_docs_scanned = 0
    t0 = time.time()
    for shard, fp in paths.items():
        if len(found) == sum(len(v) for v in needed.values()):
            break
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                n_docs_scanned += 1
                d = json.loads(line)
                text = find_text_field(d)
                if text is None:
                    continue
                first200 = text[:200]
                if first200 not in needed:
                    continue
                for i_row, role in needed[first200]:
                    if i_row not in found:
                        found[i_row] = text
        print(f"  scanned {shard:65s}  found={len(found)}/{sum(len(v) for v in needed.values())}  "
              f"docs={n_docs_scanned}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"scan done in {time.time()-t0:.0f}s — found {len(found)} matching docs")

    # Build eval prompts
    eval_records = []
    n_fail = 0
    for i_row in eval_pick:
        i_row = int(i_row)
        sid = int(acts.column("sample_idx")[i_row].as_py())
        pos = acts.column("position")[i_row].as_py()
        tok_str = acts.column("token_str")[i_row].as_py()
        if i_row not in found:
            n_fail += 1
            continue
        text = found[i_row]
        ids = tok(text, add_special_tokens=True)["input_ids"]
        if pos + 1 > len(ids):
            n_fail += 1
            continue
        decoded = tok.decode(ids[:pos + 1], skip_special_tokens=True)
        last_tok = tok.decode([ids[pos]], skip_special_tokens=False)
        prompt = _DEFAULT_INSTRUCTION.format(text=decoded)
        av_expl = res.column("explanation")[res_map[sid]].as_py() if sid in res_map else None
        eval_records.append({
            "sample_idx": sid,
            "position": pos,
            "token_str": tok_str,
            "last_tok_str": last_tok,
            "rendered_token_len": len(ids),
            "decoded_full": decoded,
            "warmstart_prompt": prompt,
            "explanation_av": av_expl,
        })
    print(f"  eval prompts: {len(eval_records)}; failures: {n_fail}")

    # Sanity stats
    n_tok_match = sum(1 for r in eval_records if r["last_tok_str"] == r["token_str"])
    decoded_lens = [len(r["decoded_full"]) for r in eval_records]
    print(f"  last-token match: {n_tok_match}/{len(eval_records)} ({100*n_tok_match/max(1,len(eval_records)):.1f}%)")
    print(f"  decoded prefix lens: mean={np.mean(decoded_lens):.0f} median={np.median(decoded_lens):.0f} "
          f"p10={np.percentile(decoded_lens,10):.0f} p90={np.percentile(decoded_lens,90):.0f} max={max(decoded_lens)}")

    # Build few-shot — keep first N that succeed
    fs_records = []
    for i_row in fs_pick:
        if len(fs_records) >= args.n_shots:
            break
        i_row = int(i_row)
        sid = int(acts.column("sample_idx")[i_row].as_py())
        pos = acts.column("position")[i_row].as_py()
        tok_str = acts.column("token_str")[i_row].as_py()
        if i_row not in found:
            continue
        text = found[i_row]
        ids = tok(text, add_special_tokens=True)["input_ids"]
        if pos + 1 > len(ids):
            continue
        decoded = tok.decode(ids[:pos + 1], skip_special_tokens=True)
        av_expl = res.column("explanation")[res_map[sid]].as_py()
        prompt = _DEFAULT_INSTRUCTION.format(text=decoded)
        response = f"<analysis>\n{av_expl}\n</analysis>"
        fs_records.append({
            "sample_idx": sid,
            "position": pos,
            "token_str": tok_str,
            "decoded_len_chars": len(decoded),
            "prompt": prompt,
            "response": response,
        })
        print(f"  shot sample_idx={sid}  decoded_len={len(decoded):4d}  av_expl_len={len(av_expl):4d}  "
              f"token={tok_str!r}")

    out = pa.table({
        "sample_idx": pa.array([r["sample_idx"] for r in eval_records], type=pa.int64()),
        "position": pa.array([r["position"] for r in eval_records], type=pa.int32()),
        "token_str": pa.array([r["token_str"] for r in eval_records]),
        "decoded_full": pa.array([r["decoded_full"] for r in eval_records]),
        "warmstart_prompt": pa.array([r["warmstart_prompt"] for r in eval_records]),
        "explanation_av": pa.array([r["explanation_av"] for r in eval_records]),
    })
    pq.write_table(out, args.out_eval, compression="zstd")
    print(f"wrote {len(eval_records)} eval prompts → {args.out_eval}")
    Path(args.out_fewshot).write_text(json.dumps(fs_records, indent=2))
    print(f"wrote {len(fs_records)} few-shot → {args.out_fewshot}")


if __name__ == "__main__":
    main()
