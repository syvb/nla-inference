"""Reconstruction-loss sweep: AV→AR round-trip on dataset activations.

For N samples drawn across the dataset's source diversity:
  1. Pick a random non-early, non-special token position.
  2. Extract the layer-K residual-stream activation from the base model.
  3. AV verbalizes the activation → explanation text.
  4. AR reconstructs an activation from the text.
  5. Record (token, position, ||v||, MSE, cos, explanation) for offline analysis.

Two phases, run sequentially so a single H100 isn't asked to hold base + AV + AR
at the same time.

Sampling:
  Default mode `--sampling diverse_shards` picks one random shard per
  top-level source directory of the dataset (e.g. arxiv_papers, github_archive,
  project_gutenberg, …) using --seed for reproducibility, interleaves them,
  shuffles within a 50k buffer. The shard list is recorded in the output
  parquet's metadata for reproducibility.

  Mode `--sampling streaming_default` is the original
  `load_dataset(..., streaming=True).shuffle(buffer_size=10_000)`. This
  silently biases sampling toward the first source alphabetically — included
  only for backward-compat reproduction of the May-10 run.

──────────────────────────────────────────────────────────────────────────────
Phase 1 — extract activations (loads BASE model only):

    python recon_loss_sweep.py extract \
        --base-model google/gemma-3-12b-it \
        --layer 32 --n 20000 --max-len 512 --batch-size 8 \
        --sampling diverse_shards --seed 0 \
        --out activations.parquet

Phase 2 — AV via SGLang + AR locally:

    python -m sglang.launch_server \
        --model-path ./nla-gemma3-12b-L32-av \
        --port 30000 --disable-radix-cache --disable-piecewise-cuda-graph \
        --mem-fraction-static 0.55 --context-length 512 --trust-remote-code

    python recon_loss_sweep.py decode \
        --activations activations.parquet \
        --av-checkpoint ./nla-gemma3-12b-L32-av \
        --ar-checkpoint ./nla-gemma3-12b-L32-ar \
        --sglang-url http://localhost:30000 \
        --ar-device cuda:0 --av-concurrency 16 \
        --out results.parquet

Smoke (no GPU, no dataset network — defaults):
    python recon_loss_sweep.py smoke
Smoke + dataset connectivity:
    python recon_loss_sweep.py smoke --check-dataset
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


# ─── Small helpers (importable for tests, no torch import at module load) ────


def find_layers_module_list(model):
    """Locate the ModuleList of transformer blocks across Qwen2/Llama/Gemma3.

    Tries common HF paths in order, including both transformers <5
    and >=5 layouts (the latter wraps Gemma3ForConditionalGeneration so
    decoder layers live under model.language_model.layers).
    """
    import torch

    paths = (
        ("model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("transformer", "h"),
        ("transformer", "layers"),
    )
    for path in paths:
        x = model
        try:
            for attr in path:
                x = getattr(x, attr)
        except AttributeError:
            continue
        if isinstance(x, torch.nn.ModuleList):
            return x
    raise RuntimeError(
        f"could not find transformer-layer ModuleList on {type(model).__name__} "
        f"(tried {paths!r}). Add this arch's path to find_layers_module_list."
    )


def get_d_model(model) -> int:
    """hidden_size, walking through .text_config for multimodal wrappers."""
    cfg = getattr(model.config, "text_config", model.config)
    return int(cfg.hidden_size)


def pick_random_position(
    input_ids,           # 1-D int tensor or list [T]
    attention_mask,      # 1-D 0/1 tensor or list [T]
    special_ids: set,
    *,
    skip_first: int,
    rng: random.Random | None = None,
) -> int:
    """Random valid token position. Returns -1 if no valid positions.

    Valid = position ≥ skip_first, attention_mask[p] == 1, token id ∉ special_ids.
    """
    rng = rng or random
    ids = list(input_ids) if not hasattr(input_ids, "tolist") else input_ids.tolist()
    mask = list(attention_mask) if not hasattr(attention_mask, "tolist") else attention_mask.tolist()
    pool = [
        p for p in range(skip_first, len(ids))
        if mask[p] == 1 and ids[p] not in special_ids
    ]
    if not pool:
        return -1
    return rng.choice(pool)


# ─── Sampling: diverse-shard interleave (the right way) ──────────────────────


def list_dataset_shards(dataset_name: str) -> dict[str, list[str]]:
    """Return {source_dir: [shard_paths]} for the dataset's jsonl.gz files.

    Uses HfApi to list repo files. Cheap (a few seconds, no downloads).
    """
    from huggingface_hub import HfApi
    api = HfApi()
    files = [f for f in api.list_repo_files(dataset_name, repo_type="dataset")
             if f.endswith(".jsonl.gz") or f.endswith(".jsonl") or f.endswith(".parquet")]
    if not files:
        raise RuntimeError(
            f"No jsonl(.gz)/parquet shards found in {dataset_name}. "
            f"Did the dataset layout change? Inspect with HfApi.list_repo_files."
        )
    by_source: dict[str, list[str]] = defaultdict(list)
    for f in files:
        # Top-level dir is the source. Files at repo root go into "_root_".
        parts = f.split("/", 1)
        src = parts[0] if len(parts) == 2 else "_root_"
        by_source[src].append(f)
    return {k: sorted(v) for k, v in sorted(by_source.items())}


def select_diverse_shards(by_source: dict[str, list[str]], seed: int,
                          shards_per_source: int = 1) -> list[str]:
    """Deterministically pick `shards_per_source` random shards per source.

    Returns the flat list of shard paths in deterministic order.
    """
    rng = random.Random(seed)
    selected: list[str] = []
    for src in sorted(by_source):
        shards = list(by_source[src])
        rng.shuffle(shards)
        selected.extend(shards[:shards_per_source])
    return selected


def diverse_shard_iter(dataset_name: str, seed: int,
                       shards_per_source: int = 1,
                       buffer_size: int = 50_000) -> Iterable[str]:
    """Yield text strings drawn across the dataset's source diversity.

    For each top-level source dir, picks `shards_per_source` random shard(s)
    (deterministic on seed). Loads each as a streaming dataset, interleaves
    them with `interleave_datasets`, then shuffles within a buffer.

    The selected shard list is logged so the run is reproducible.
    """
    from datasets import interleave_datasets, load_dataset

    by_source = list_dataset_shards(dataset_name)
    selected = select_diverse_shards(by_source, seed, shards_per_source)
    print(f"[sample] using {len(selected)} shards from "
          f"{len(by_source)} source dirs (seed={seed}):")
    for s in selected:
        print(f"  - {s}")

    sub_dss = [load_dataset(dataset_name, data_files=s, streaming=True,
                            split="train") for s in selected]
    iled = interleave_datasets(sub_dss, seed=seed,
                               stopping_strategy="all_exhausted")
    iled = iled.shuffle(seed=seed, buffer_size=buffer_size)

    diverse_shard_iter.last_selection = selected   # introspection / metadata

    for ex in iled:
        text = ex.get("text") or ex.get("content")
        if isinstance(text, str) and text.strip():
            yield text


def streaming_default_iter(dataset_name: str, split: str, seed: int,
                            buffer_size: int = 10_000) -> Iterable[str]:
    """The OLD biased sampler — kept for reproducibility of the May-10 run.

    HF streaming reads shards alphabetically; a 10k-row buffer can't escape a
    single multi-GB shard, so this effectively samples within whatever source
    sorts alphabetically first that has its shard not skipped by the seed.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    for ex in ds:
        text = ex.get("text") or ex.get("content")
        if isinstance(text, str) and text.strip():
            yield text


def stream_samples(dataset_name: str, split: str, seed: int,
                   sampling: str = "diverse_shards") -> Iterable[str]:
    """Dispatch to the chosen sampling method."""
    if sampling == "diverse_shards":
        return diverse_shard_iter(dataset_name, seed)
    if sampling == "streaming_default":
        return streaming_default_iter(dataset_name, split, seed)
    raise ValueError(f"unknown sampling mode: {sampling}")


# ─── Phase 1: extract ────────────────────────────────────────────────────────


def cmd_extract(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # cuDNN's SDPA backend errors out on Gemma-3 ("No valid execution plans
    # built") with driver 580 + cuDNN 9.x. Disable it; flash/mem-efficient
    # backends still work and are plenty fast for forward-only.
    torch.backends.cuda.enable_cudnn_sdp(False)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[extract] loading {args.base_model} (bf16, device_map={args.device_map})")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    layers = find_layers_module_list(model)
    if not (0 <= args.layer < len(layers)):
        sys.exit(f"--layer {args.layer} out of range [0,{len(layers)})")
    target_block = layers[args.layer]
    d_model = get_d_model(model)
    print(f"[extract] hooking layers[{args.layer}] of {len(layers)} "
          f"({type(target_block).__name__}); d_model={d_model}")

    captured: dict[str, "torch.Tensor"] = {}

    def hook(_, _ins, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h

    handle = target_block.register_forward_hook(hook)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    special_ids = set(tok.all_special_ids or [])

    # Vectors as fp32 (Gemma residual elements exceed fp16 max for ~60% of
    # samples, so fp16 storage destroys the data).
    schema = pa.schema([
        pa.field("sample_idx", pa.int64()),
        pa.field("token_id", pa.int64()),
        pa.field("token_str", pa.string()),
        pa.field("position", pa.int32()),
        pa.field("seq_len", pa.int32()),
        pa.field("vec_norm", pa.float32()),
        pa.field("text_preview", pa.string()),
        pa.field("activation", pa.list_(pa.float32(), d_model)),
    ])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(out_path), schema, compression="zstd")

    rows: dict[str, list] = {k: [] for k in schema.names}
    n_written = 0
    n_attempted = 0
    t0 = time.time()

    src = stream_samples(args.dataset, args.split, args.seed,
                         sampling=args.sampling)

    while n_written < args.n:
        texts: list[str] = []
        while len(texts) < args.batch_size and n_written + len(texts) < args.n:
            try:
                texts.append(next(src))
            except StopIteration:
                break
        if not texts:
            break

        enc = tok(
            texts, return_tensors="pt", truncation=True,
            max_length=args.max_len, padding=True,
        ).to(model.device)

        with torch.no_grad():
            model(input_ids=enc["input_ids"],
                  attention_mask=enc["attention_mask"],
                  use_cache=False)
        h = captured["h"]
        assert h.shape[-1] == d_model, (
            f"hooked tensor d={h.shape[-1]} != d_model={d_model}; "
            f"target_block produced an unexpected shape — wrong layer index?"
        )

        ids_cpu = enc["input_ids"].cpu()
        mask_cpu = enc["attention_mask"].cpu()
        for i in range(len(texts)):
            n_attempted += 1
            p = pick_random_position(
                ids_cpu[i], mask_cpu[i], special_ids,
                skip_first=args.skip_first, rng=rng,
            )
            if p < 0:
                continue
            v_fp32 = h[i, p].detach().to(torch.float32).cpu().numpy()
            tok_id = int(ids_cpu[i, p].item())
            seq_len = int(mask_cpu[i].sum().item())
            rows["sample_idx"].append(n_written)
            rows["token_id"].append(tok_id)
            rows["token_str"].append(tok.decode([tok_id]))
            rows["position"].append(int(p))
            rows["seq_len"].append(seq_len)
            rows["vec_norm"].append(float(np.linalg.norm(v_fp32)))
            rows["text_preview"].append(texts[i][:200])
            rows["activation"].append(v_fp32.tolist())
            n_written += 1
            if n_written >= args.n:
                break

        if len(rows["sample_idx"]) >= args.flush_every:
            writer.write_table(pa.table(rows, schema=schema))
            rows = {k: [] for k in schema.names}
            elapsed = time.time() - t0
            rate = n_written / max(elapsed, 1e-6)
            eta = (args.n - n_written) / max(rate, 1e-6)
            print(f"[extract] {n_written}/{args.n}  attempted={n_attempted}  "
                  f"{rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

    if rows["sample_idx"]:
        writer.write_table(pa.table(rows, schema=schema))
    writer.close()
    handle.remove()

    # Sidecar JSON: shard list + run metadata for reproducibility.
    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "sampling": args.sampling,
        "seed": args.seed,
        "n_target": args.n,
        "n_written": n_written,
        "max_len": args.max_len,
        "skip_first": args.skip_first,
        "base_model": args.base_model,
        "layer": args.layer,
        "selected_shards": list(getattr(diverse_shard_iter,
                                        "last_selection", []) or []),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[extract] done. wrote {n_written} rows to {out_path} "
          f"in {(time.time()-t0)/60:.1f} min")
    print(f"[extract] metadata sidecar: {meta_path}")


# ─── Phase 2: decode (AV via SGLang + AR locally) ────────────────────────────


def cmd_decode(args):
    import httpx
    import orjson
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nla_inference import EXPLANATION_RE, NLAClient, NLACritic

    rng = random.Random(args.seed)  # noqa: F841

    av_client = NLAClient(args.av_checkpoint, sglang_url=args.sglang_url)
    critic = NLACritic(args.ar_checkpoint, device=args.ar_device)

    pf = pq.ParquetFile(args.activations)
    n_total = pf.metadata.num_rows
    # Old fp16 schema used "activation_fp16" — read either name.
    act_col = ("activation" if "activation" in pf.schema_arrow.names
               else "activation_fp16")
    act_field = pf.schema_arrow.field(act_col)
    d_act = act_field.type.list_size
    d_av = av_client.cfg.d_model
    if d_act != d_av:
        sys.exit(f"d_model mismatch: activations={d_act}, AV checkpoint={d_av}.")
    print(f"[decode] {n_total} activations  d_model={d_act}  "
          f"(reading column '{act_col}')")

    out_schema = pa.schema([
        pa.field("sample_idx", pa.int64()),
        pa.field("token_id", pa.int64()),
        pa.field("token_str", pa.string()),
        pa.field("position", pa.int32()),
        pa.field("seq_len", pa.int32()),
        pa.field("vec_norm", pa.float32()),
        pa.field("text_preview", pa.string()),
        pa.field("explanation", pa.string()),
        pa.field("raw_av_text", pa.string()),
        pa.field("av_parsed", pa.bool_()),
        pa.field("mse_nrm", pa.float32()),
        pa.field("cos", pa.float32()),
        pa.field("activation", pa.list_(pa.float32(), d_act)),
        pa.field("recon", pa.list_(pa.float32(), d_act)),
    ])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(args.out), out_schema, compression="zstd")

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "skip_special_tokens": False,
    }

    async def run():
        # Bigger pool than the default (100/20) — av_concurrency can be 16+
        # and we want headroom for retries that briefly hold both the new and
        # the dying connection.
        limits = httpx.Limits(max_connections=args.av_concurrency * 4,
                              max_keepalive_connections=args.av_concurrency)
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(args.av_timeout),
            limits=limits,
        )
        sem = asyncio.Semaphore(args.av_concurrency)

        async def call_av(vec_np: np.ndarray) -> str:
            embeds_np, _ = av_client._build_embeds(
                torch.as_tensor(vec_np, dtype=torch.float32), prompt_content=None
            )
            body = orjson.dumps(
                {"input_embeds": embeds_np, "sampling_params": sampling_params},
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
            # Transient httpx.ReadError/RemoteProtocolError can hit at
            # 10+ concurrent requests vs. sglang. Cheap to retry.
            last_exc: Exception | None = None
            for attempt in range(4):
                try:
                    async with sem:
                        resp = await client.post(
                            f"{av_client.sglang_url}/generate",
                            content=body,
                            headers={"Content-Type": "application/json"},
                        )
                    resp.raise_for_status()
                    out = resp.json()
                    return (out[0] if isinstance(out, list) else out)["text"]
                except (httpx.ReadError, httpx.RemoteProtocolError,
                        httpx.ConnectError, httpx.PoolTimeout) as e:
                    last_exc = e
                    await asyncio.sleep(0.5 * (2 ** attempt))
            raise RuntimeError(f"AV call failed after 4 retries: {last_exc!r}")

        rows: dict[str, list] = {k: [] for k in out_schema.names}
        n_done = 0
        t0 = time.time()

        for batch in pf.iter_batches(batch_size=args.batch_size, columns=[
            "sample_idx", "token_id", "token_str", "position", "seq_len",
            "vec_norm", "text_preview", act_col,
        ]):
            d = batch.to_pydict()
            vecs = [np.asarray(v, dtype=np.float32) for v in d[act_col]]

            raw_texts = await asyncio.gather(*(call_av(v) for v in vecs))

            for i, (v, raw) in enumerate(zip(vecs, raw_texts)):
                m = EXPLANATION_RE.search(raw)
                if m is None:
                    expl = raw.strip()
                    parsed = False
                else:
                    expl = m.group(1).strip()
                    parsed = True

                pred = critic.reconstruct(expl)
                gold = torch.as_tensor(v, dtype=torch.float32)
                pred_n = pred / pred.norm().clamp_min(1e-12) * critic.mse_scale
                gold_n = gold / gold.norm().clamp_min(1e-12) * critic.mse_scale
                mse = float(((pred_n - gold_n) ** 2).mean().item())
                cos = float((pred_n @ gold_n
                             / (pred_n.norm() * gold_n.norm())).item())

                rows["sample_idx"].append(d["sample_idx"][i])
                rows["token_id"].append(d["token_id"][i])
                rows["token_str"].append(d["token_str"][i])
                rows["position"].append(d["position"][i])
                rows["seq_len"].append(d["seq_len"][i])
                rows["vec_norm"].append(d["vec_norm"][i])
                rows["text_preview"].append(d["text_preview"][i])
                rows["explanation"].append(expl)
                rows["raw_av_text"].append(raw)
                rows["av_parsed"].append(parsed)
                rows["mse_nrm"].append(mse)
                rows["cos"].append(cos)
                rows["activation"].append(
                    d[act_col][i] if act_col == "activation"
                    else np.asarray(d[act_col][i], dtype=np.float32).tolist()
                )
                rows["recon"].append(
                    pred.to(torch.float32).cpu().numpy().tolist()
                )
                n_done += 1

            if len(rows["sample_idx"]) >= args.flush_every:
                writer.write_table(pa.table(rows, schema=out_schema))
                rows = {k: [] for k in out_schema.names}
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-6)
                eta = (n_total - n_done) / max(rate, 1e-6)
                print(f"[decode] {n_done}/{n_total}  {rate:.1f}/s  "
                      f"eta={eta/60:.1f}min", flush=True)

        if rows["sample_idx"]:
            writer.write_table(pa.table(rows, schema=out_schema))
        await client.aclose()

    asyncio.run(run())
    writer.close()
    print(f"[decode] done. wrote {args.out}")


# ─── Smoke ───────────────────────────────────────────────────────────────────


def cmd_smoke(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    print("[smoke] pick_random_position …")
    rng = random.Random(0)
    ids = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 200, 201, 202, 203, 204]
    mask = [1] * 15
    p = pick_random_position(ids, mask, special_ids={204}, skip_first=10, rng=rng)
    assert 10 <= p < 14, f"got {p}"
    print("    OK")

    print("[smoke] parquet schema round-trip …")
    schema = pa.schema([
        pa.field("sample_idx", pa.int64()),
        pa.field("activation", pa.list_(pa.float32(), 8)),
    ])
    rows = {"sample_idx": [0, 1],
            "activation": [list(np.arange(8, dtype=np.float32)),
                           list(np.arange(8, dtype=np.float32)[::-1])]}
    tbl = pa.table(rows, schema=schema)
    tmp = Path("/tmp/_recon_smoke.parquet")
    pq.write_table(tbl, tmp, compression="zstd")
    back = pq.read_table(tmp)
    assert back.num_rows == 2
    print("    OK")

    if args.check_dataset:
        print(f"[smoke] diverse_shards probe (seed=0)…")
        try:
            it = stream_samples(args.dataset, args.split, 0,
                                sampling="diverse_shards")
            for i in range(args.n):
                t = next(it)
                print(f"    [{i}] {t[:80]!r}")
        except Exception as e:
            print(f"    FAILED: {e}")
            return 1
    else:
        print("[smoke] skipping dataset probe (pass --check-dataset to test)")

    print("[smoke] done")
    return 0


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    pe = sp.add_parser("extract", help="phase 1: base model → activations parquet")
    pe.add_argument("--base-model", required=True)
    pe.add_argument("--layer", type=int, required=True)
    pe.add_argument("--n", type=int, default=20_000)
    pe.add_argument("--max-len", type=int, default=512)
    pe.add_argument("--batch-size", type=int, default=8)
    pe.add_argument("--skip-first", type=int, default=10)
    pe.add_argument("--dataset", default="common-pile/comma_v0.1_training_dataset")
    pe.add_argument("--split", default="train")
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--sampling", default="diverse_shards",
                    choices=["diverse_shards", "streaming_default"],
                    help="diverse_shards = pick 1 random shard per source dir "
                         "(reproducible via seed). streaming_default = the old "
                         "biased buffered shuffle.")
    pe.add_argument("--device-map", default="auto")
    pe.add_argument("--out", required=True)
    pe.add_argument("--flush-every", type=int, default=512)
    pe.set_defaults(func=cmd_extract)

    pd_ = sp.add_parser("decode")
    pd_.add_argument("--activations", required=True)
    pd_.add_argument("--av-checkpoint", required=True)
    pd_.add_argument("--ar-checkpoint", required=True)
    pd_.add_argument("--sglang-url", default="http://localhost:30000")
    pd_.add_argument("--ar-device", default="cuda:0")
    pd_.add_argument("--av-concurrency", type=int, default=16)
    pd_.add_argument("--av-timeout", type=float, default=120.0)
    pd_.add_argument("--temperature", type=float, default=0.0)
    pd_.add_argument("--max-new-tokens", type=int, default=200)
    pd_.add_argument("--batch-size", type=int, default=64)
    pd_.add_argument("--flush-every", type=int, default=512)
    pd_.add_argument("--seed", type=int, default=0)
    pd_.add_argument("--out", required=True)
    pd_.set_defaults(func=cmd_decode)

    ps = sp.add_parser("smoke")
    ps.add_argument("--check-dataset", action="store_true")
    ps.add_argument("--n", type=int, default=5)
    ps.add_argument("--dataset", default="common-pile/comma_v0.1_training_dataset")
    ps.add_argument("--split", default="train")
    ps.set_defaults(func=cmd_smoke)

    args = p.parse_args(argv)
    rc = args.func(args)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
