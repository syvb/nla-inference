"""Reconstruction-loss sweep: AV→AR round-trip on dataset activations.

For N random samples from common-pile/comma_v0.1_training_dataset:
  1. Pick a random non-early, non-special token position.
  2. Extract the layer-K residual-stream activation from the base model.
  3. AV verbalizes the activation → explanation text.
  4. AR reconstructs an activation from the text.
  5. Record (token, position, ||v||, MSE, cos, explanation) for offline analysis.

Two phases, run sequentially so a single H100 isn't asked to hold base + AV + AR
at the same time.

──────────────────────────────────────────────────────────────────────────────
Phase 1 — extract activations (loads BASE model only):

    python recon_loss_sweep.py extract \
        --base-model google/gemma-3-27b-it \
        --layer 41 --n 100000 --max-len 512 --batch-size 4 \
        --out activations.parquet

Phase 2 — AV via SGLang + AR locally:

    # In another shell, launch SGLang AV server. Use mem-fraction-static=0.55
    # so AR has room to load on the same H100; if AV is on a different machine,
    # bump it to 0.85.
    python -m sglang.launch_server \
        --model-path ./nla-gemma3-27b-L41-av \
        --port 30000 --disable-radix-cache \
        --mem-fraction-static 0.55 --trust-remote-code

    python recon_loss_sweep.py decode \
        --activations activations.parquet \
        --av-checkpoint ./nla-gemma3-27b-L41-av \
        --ar-checkpoint ./nla-gemma3-27b-L41-ar \
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
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


# ─── Small helpers (importable for tests, no torch import at module load) ────


def find_layers_module_list(model):
    """Locate the ModuleList of transformer blocks across Qwen2/Llama/Gemma3.

    Tries the common HF paths in order. Gemma3ForConditionalGeneration nests
    text decoder layers under .language_model; plain CausalLM under .model.
    """
    import torch

    paths = (
        ("model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        # transformers >= 5.x wraps Gemma3ForConditionalGeneration so the
        # text-decoder ModuleList lives under model.language_model.layers.
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


def stream_samples(dataset_name: str, split: str, seed: int) -> Iterable[str]:
    """Yield non-empty `text` strings from the streaming dataset, shuffled.

    Robust to schemas that name the field differently — falls back to "content".
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for ex in ds:
        text = ex.get("text") or ex.get("content")
        if isinstance(text, str) and text.strip():
            yield text


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

    # The hook captures the residual-stream output of block K. HF hidden_states
    # convention: hidden_states[K+1] = output of layer K. The block returns
    # either a Tensor or (Tensor, ...); we take [0] in the tuple case.
    captured: dict[str, torch.Tensor] = {}

    def hook(_, _ins, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h

    handle = target_block.register_forward_hook(hook)

    # Pad-on-the-left would shift positions; HF default is right-padding which
    # keeps real positions at small indices. We keep right-padding and use the
    # attention mask to bound the "valid" range.
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    special_ids = set(tok.all_special_ids or [])

    # Fixed-size list keeps reads ~10× faster than variable list for the vector
    # column; d_model is known here. fp32 (not fp16) — Gemma-3 residual-stream
    # elements routinely exceed fp16's 65504 max (~64% of layer-32 vectors had
    # at least one inf element when stored as fp16), which destroys the data.
    # bf16-source → fp32 is a lossless upcast.
    schema = pa.schema([
        pa.field("sample_idx", pa.int64()),
        pa.field("token_id", pa.int64()),
        pa.field("token_str", pa.string()),
        pa.field("position", pa.int32()),
        pa.field("seq_len", pa.int32()),
        pa.field("vec_norm", pa.float32()),     # raw L2-norm of activation
        pa.field("text_preview", pa.string()),  # first 200 chars
        pa.field("activation", pa.list_(pa.float32(), d_model)),
    ])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(out_path), schema, compression="zstd")

    rows: dict[str, list] = {k: [] for k in schema.names}
    n_written = 0
    n_attempted = 0
    t0 = time.time()

    src = stream_samples(args.dataset, args.split, args.seed)

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
        h = captured["h"]  # [B, T, d]
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
            v_fp32 = h[i, p].detach().to(torch.float32).cpu().numpy()  # [d]
            tok_id = int(ids_cpu[i, p].item())
            seq_len = int(mask_cpu[i].sum().item())
            rows["sample_idx"].append(n_written)
            rows["token_id"].append(tok_id)
            # decode([id]) on a single id is a fast path; for byte-level BPE it
            # may yield the visible piece (e.g. " the" with leading space) which
            # is exactly what we want for downstream grouping.
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
    print(f"[extract] done. wrote {n_written} rows to {out_path} "
          f"in {(time.time()-t0)/60:.1f} min")


# ─── Phase 2: decode (AV via SGLang + AR locally) ────────────────────────────


def cmd_decode(args):
    import httpx
    import orjson
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    # nla_inference is a sibling file in this repo.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nla_inference import EXPLANATION_RE, NLAClient, NLACritic

    rng = random.Random(args.seed)  # noqa: F841 — kept for future stochastic ops

    av_client = NLAClient(args.av_checkpoint, sglang_url=args.sglang_url)
    critic = NLACritic(args.ar_checkpoint, device=args.ar_device)

    # Cross-check d_model between the activations file and the loaded checkpoints.
    pf = pq.ParquetFile(args.activations)
    n_total = pf.metadata.num_rows
    # Old fp16 schema used "activation_fp16" — read either name.
    act_col = "activation" if "activation" in pf.schema_arrow.names else "activation_fp16"
    act_field = pf.schema_arrow.field(act_col)
    d_act = act_field.type.list_size
    d_av = av_client.cfg.d_model
    if d_act != d_av:
        sys.exit(f"d_model mismatch: activations={d_act}, AV checkpoint={d_av}. "
                 f"Wrong checkpoint for these activations?")
    print(f"[decode] {n_total} activations  d_model={d_act}")

    out_schema = pa.schema([
        pa.field("sample_idx", pa.int64()),
        pa.field("token_id", pa.int64()),
        pa.field("token_str", pa.string()),
        pa.field("position", pa.int32()),
        pa.field("seq_len", pa.int32()),
        pa.field("vec_norm", pa.float32()),
        pa.field("text_preview", pa.string()),
        pa.field("explanation", pa.string()),
        pa.field("raw_av_text", pa.string()),    # full AV gen, pre-tag-extraction
        pa.field("av_parsed", pa.bool_()),       # whether <explanation> tags found
        pa.field("mse_nrm", pa.float32()),
        pa.field("cos", pa.float32()),
        # Vectors as fp32 (Gemma residual elements exceed fp16 max for ~60% of
        # samples, so fp16 storage destroys the data).
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
        # Larger pool than the default (100/20) — av_concurrency can be 16+
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
            # _build_embeds is sync (CPU torch); the embed lookup is
            # ~100 token IDs through a bf16 table on CPU — a few ms, fine in
            # the event loop. The HTTP wait is the actual bottleneck.
            embeds_np, _ = av_client._build_embeds(
                torch.as_tensor(vec_np, dtype=torch.float32), prompt_content=None
            )
            body = orjson.dumps(
                {"input_embeds": embeds_np, "sampling_params": sampling_params},
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
            # Transient httpx.ReadError / RemoteProtocolError can hit at
            # 10+ concurrent requests vs. sglang. Cheap to retry — sglang
            # /generate is idempotent.
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

            # AV in parallel (network-bound).
            raw_texts = await asyncio.gather(*(call_av(v) for v in vecs))

            for i, (v, raw) in enumerate(zip(vecs, raw_texts)):
                m = EXPLANATION_RE.search(raw)
                if m is None:
                    expl = raw.strip()
                    parsed = False
                else:
                    expl = m.group(1).strip()
                    parsed = True

                # AR forward — single short prompt, fast on GPU.
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
                rows["activation"].append(d[act_col][i] if act_col == "activation"
                                          else np.asarray(d[act_col][i], dtype=np.float32).tolist())
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


# ─── Smoke (no GPU, no remote services) ──────────────────────────────────────


def cmd_smoke(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    print("[smoke] pick_random_position …")
    rng = random.Random(0)
    ids = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 200, 201, 202, 203, 204]
    mask = [1] * 15
    p = pick_random_position(ids, mask, special_ids={204}, skip_first=10, rng=rng)
    assert 10 <= p < 14, f"got {p}"
    p_none = pick_random_position(
        ids, mask, special_ids=set(range(100, 250)), skip_first=10, rng=rng,
    )
    assert p_none == -1
    p_padded = pick_random_position(
        ids, [1] * 12 + [0] * 3, special_ids=set(), skip_first=10, rng=rng,
    )
    assert 10 <= p_padded < 12
    print("    OK")

    print("[smoke] parquet schema round-trip (fixed-size float16 list) …")
    d_model = 8
    schema = pa.schema([
        pa.field("sample_idx", pa.int64()),
        pa.field("token_id", pa.int64()),
        pa.field("activation_fp16", pa.list_(pa.float16(), d_model)),
    ])
    rows = {
        "sample_idx": [0, 1],
        "token_id": [42, 43],
        "activation_fp16": [
            np.arange(d_model, dtype=np.float16).tolist(),
            np.arange(d_model, dtype=np.float16).tolist()[::-1],
        ],
    }
    tbl = pa.table(rows, schema=schema)
    tmp = Path("/tmp/_recon_smoke.parquet")
    pq.write_table(tbl, tmp, compression="zstd")
    back = pq.read_table(tmp)
    assert back.num_rows == 2
    assert back.schema.field("activation_fp16").type.list_size == d_model
    print(f"    OK ({tmp.stat().st_size} B)")

    print("[smoke] importing nla_inference (no model load) …")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nla_inference  # noqa: F401
    print("    OK")

    if args.check_dataset:
        print(f"[smoke] streaming first {args.n} samples from "
              f"{args.dataset!r} (split={args.split!r}) …")
        try:
            from datasets import load_dataset
            ds = load_dataset(args.dataset, split=args.split, streaming=True)
            it = iter(ds)
            for i in range(args.n):
                ex = next(it)
                if i == 0:
                    print(f"    fields: {list(ex.keys())}")
                text = ex.get("text") or ex.get("content") or ""
                print(f"    [{i}] len={len(text)}  preview={text[:80]!r}")
        except Exception as e:
            print(f"    FAILED: {e}")
            return 1
    else:
        print("[smoke] skipping dataset connectivity (pass --check-dataset to test)")

    print("[smoke] done")
    return 0


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    # extract
    pe = sp.add_parser("extract", help="phase 1: base model → activations parquet")
    pe.add_argument("--base-model", required=True,
                    help="HF id, e.g. google/gemma-3-27b-it")
    pe.add_argument("--layer", type=int, required=True,
                    help="layer index K — output of layers[K] is captured "
                         "(= HF hidden_states[K+1]). E.g. 41 for Gemma-3-27B.")
    pe.add_argument("--n", type=int, default=100_000)
    pe.add_argument("--max-len", type=int, default=512)
    pe.add_argument("--batch-size", type=int, default=4)
    pe.add_argument("--skip-first", type=int, default=10,
                    help="Skip first N positions (early-sequence positions are "
                         "noisy per README — residual stream hasn't accumulated).")
    pe.add_argument("--dataset", default="common-pile/comma_v0.1_training_dataset")
    pe.add_argument("--split", default="train")
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--device-map", default="auto")
    pe.add_argument("--out", required=True)
    pe.add_argument("--flush-every", type=int, default=512,
                    help="Rows buffered before each parquet write.")
    pe.set_defaults(func=cmd_extract)

    # decode
    pd_ = sp.add_parser("decode",
                        help="phase 2: activations parquet → AV (SGLang) → AR → results parquet")
    pd_.add_argument("--activations", required=True,
                     help="Path to phase-1 output.")
    pd_.add_argument("--av-checkpoint", required=True,
                     help="HF-format AV (verbalizer) dir with nla_meta.yaml. "
                          "Same path SGLang is serving.")
    pd_.add_argument("--ar-checkpoint", required=True,
                     help="HF-format AR (reconstructor) dir.")
    pd_.add_argument("--sglang-url", default="http://localhost:30000")
    pd_.add_argument("--ar-device", default="cuda:0")
    pd_.add_argument("--av-concurrency", type=int, default=8,
                     help="Max in-flight AV HTTP requests. SGLang's continuous "
                          "batcher packs server-side; 8–16 is usually plenty.")
    pd_.add_argument("--av-timeout", type=float, default=120.0)
    pd_.add_argument("--temperature", type=float, default=0.0,
                     help="0.0 = greedy, reproducible. 0.7+ for sampling.")
    pd_.add_argument("--max-new-tokens", type=int, default=200)
    pd_.add_argument("--batch-size", type=int, default=64,
                     help="Rows read from activations parquet per chunk; "
                          "also the parallel AV-call window per chunk.")
    pd_.add_argument("--flush-every", type=int, default=512)
    pd_.add_argument("--seed", type=int, default=0)
    pd_.add_argument("--out", required=True)
    pd_.set_defaults(func=cmd_decode)

    # smoke
    ps = sp.add_parser("smoke", help="CPU-only sanity checks")
    ps.add_argument("--check-dataset", action="store_true",
                    help="Also stream a few samples from the HF dataset "
                         "(requires network).")
    ps.add_argument("--n", type=int, default=3)
    ps.add_argument("--dataset", default="common-pile/comma_v0.1_training_dataset")
    ps.add_argument("--split", default="train")
    ps.set_defaults(func=cmd_smoke)

    args = p.parse_args(argv)
    rc = args.func(args)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
