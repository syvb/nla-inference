"""Chat-aware activation extract + decode for regenerated WildChat conversations.

Forked from recon_loss_sweep.py but adapted for chat data:
  - Reads regen JSONL (one conv per line) instead of streaming raw text
  - Applies Gemma chat template to each conversation
  - Tracks 5-way role per token: bos / user_content / asst_content / role_marker / end_of_turn
  - Picks ANY random token (including specials), records role
  - Skips the similarity computation in decode (no cos / mse_nrm columns)
  - Stores chars_before / chars_after (50 char window) at extract time
  - max_len = 2048 to capture more of each conversation

Output schema (extract):
  sample_idx, conv_hash, n_user_turns, n_asst_turns,
  token_id, token_str, position, unpadded_pos, role, chars_before, chars_after,
  seq_len, vec_norm, activation (float32[d_model])

Output schema (decode):
  ...all of above plus explanation, raw_av_text, av_parsed, recon (float32[d_model])

Usage:
    python recon_chat.py extract \\
      --base-model google/gemma-3-12b-it --layer 32 \\
      --regen-jsonl regen_wildchat_20k.jsonl \\
      --n 20000 --max-len 2048 --batch-size 2 \\
      --out activations.parquet

    python recon_chat.py decode \\
      --activations activations.parquet \\
      --av-checkpoint nla-av --ar-checkpoint nla-ar \\
      --sglang-url http://localhost:30000 \\
      --av-concurrency 16 \\
      --out results.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


# Roles for the 5-way classification.
ROLE_BOS = "bos"
ROLE_USER = "user_content"
ROLE_ASST = "asst_content"
ROLE_MARKER = "role_marker"
ROLE_EOT = "end_of_turn"


def find_layers_module_list(model):
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
    raise RuntimeError(f"can't find layers on {type(model).__name__}")


def get_d_model(model) -> int:
    cfg = getattr(model.config, "text_config", model.config)
    return int(cfg.hidden_size)


def format_and_tokenize_chat(messages: list[dict], tokenizer, max_len: int):
    """Apply Gemma chat template, tokenize, return (input_ids, role_per_token).

    Role labels are computed by walking the token sequence and identifying
    the special markers Gemma uses. Gemma-3 chat template:
        <bos><start_of_turn>user
        {content}<end_of_turn>
        <start_of_turn>model
        {content}<end_of_turn>
        ...
    Token ids for Gemma-3:
        bos = 2
        <start_of_turn> = 106
        <end_of_turn> = 107
    """
    # apply_chat_template returns BatchEncoding[Encoding] in transformers 5.x
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        truncation=True, max_length=max_len,
    )
    # Flatten — same shim as nla_inference._flatten_chat_ids
    if isinstance(out, list) and out and isinstance(out[0], int):
        ids = out
    else:
        first = out[0]
        if hasattr(first, "ids"):
            ids = list(first.ids)
        elif isinstance(first, list) and first and isinstance(first[0], int):
            ids = list(first)
        else:
            raise TypeError(f"unexpected apply_chat_template return: {type(out)}")

    # ID constants for Gemma-3
    BOS_ID = tokenizer.bos_token_id or 2
    SOT_ID = tokenizer.convert_tokens_to_ids("<start_of_turn>")
    EOT_ID = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    NEWLINE_ID = tokenizer.encode("\n", add_special_tokens=False)
    NEWLINE_ID = NEWLINE_ID[0] if NEWLINE_ID else -1

    # Walk through tokens and assign roles
    roles = [None] * len(ids)
    state = "init"   # init → seen_sot → in_role_marker → in_content
    current_role = None    # "user" or "model"

    i = 0
    while i < len(ids):
        tid = ids[i]
        if tid == BOS_ID:
            roles[i] = ROLE_BOS
            i += 1
            continue
        if tid == SOT_ID:
            # Next token(s) are the role marker until we hit a newline
            roles[i] = ROLE_MARKER
            j = i + 1
            role_pieces = []
            while j < len(ids) and ids[j] != NEWLINE_ID:
                roles[j] = ROLE_MARKER
                role_pieces.append(ids[j])
                j += 1
            # Also mark the newline as marker (it terminates the role line)
            if j < len(ids) and ids[j] == NEWLINE_ID:
                roles[j] = ROLE_MARKER
                j += 1
            # Decode the role tokens
            role_str = tokenizer.decode(role_pieces).strip()
            current_role = ROLE_USER if role_str == "user" else ROLE_ASST
            i = j
            continue
        if tid == EOT_ID:
            roles[i] = ROLE_EOT
            current_role = None
            i += 1
            continue
        # Otherwise — content of current role
        if current_role is None:
            # Stray tokens between turns; classify as marker for safety
            roles[i] = ROLE_MARKER
        else:
            roles[i] = current_role
        i += 1

    return ids, roles


def pick_random_position(input_ids, *, rng: random.Random) -> int:
    """Random valid position 0..len-1. ANY token, including specials/markers."""
    if not input_ids:
        return -1
    return rng.randrange(len(input_ids))


# ─── Phase 1: extract ────────────────────────────────────────────────────────

def cmd_extract(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.enable_cudnn_sdp(False)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[extract] loading {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map=args.device_map, trust_remote_code=True,
    )
    model.eval()

    layers = find_layers_module_list(model)
    target_block = layers[args.layer]
    d_model = get_d_model(model)
    print(f"[extract] layer {args.layer}/{len(layers)}  d_model={d_model}")

    captured: dict[str, "torch.Tensor"] = {}

    def hook(_, _ins, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = target_block.register_forward_hook(hook)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    schema = pa.schema([
        pa.field("sample_idx", pa.int64()),
        pa.field("conv_hash", pa.string()),
        pa.field("n_user_turns", pa.int32()),
        pa.field("n_asst_turns", pa.int32()),
        pa.field("token_id", pa.int64()),
        pa.field("token_str", pa.string()),
        pa.field("position", pa.int32()),
        pa.field("unpadded_pos", pa.int32()),
        pa.field("role", pa.string()),
        pa.field("chars_before", pa.string()),
        pa.field("chars_after", pa.string()),
        pa.field("seq_len", pa.int32()),
        pa.field("vec_norm", pa.float32()),
        pa.field("activation", pa.list_(pa.float32(), d_model)),
    ])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(out_path), schema, compression="zstd")

    rows: dict[str, list] = {k: [] for k in schema.names}
    n_written = 0
    t0 = time.time()

    # Buffer for batching
    pending: list[tuple] = []  # (sample_idx_local, conv_hash, ids, roles, chars_b, chars_a, pos, n_u, n_a)

    def flush_batch(batch):
        """Run forward on a batch of variable-length sequences (left-pad to longest)."""
        if not batch:
            return
        max_t = max(len(ids) for _, _, ids, _, _, _, _, _, _ in batch)
        # Gemma uses left-padding by default
        pad_id = tok.pad_token_id
        b = len(batch)
        input_ids = torch.full((b, max_t), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_t), dtype=torch.long)
        for bi, (_, _, ids, _, _, _, _, _, _) in enumerate(batch):
            seq_len = len(ids)
            input_ids[bi, max_t - seq_len:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[bi, max_t - seq_len:] = 1
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        h = captured["h"]
        assert h.shape[-1] == d_model

        nonlocal n_written
        for bi, (loc_idx, conv_hash, ids, roles, chars_b, chars_a,
                 unpadded_pos, n_u, n_a) in enumerate(batch):
            padded_pos = (max_t - len(ids)) + unpadded_pos
            v_fp32 = h[bi, padded_pos].detach().to(torch.float32).cpu().numpy()
            tok_id = ids[unpadded_pos]
            seq_len = len(ids)
            rows["sample_idx"].append(n_written)
            rows["conv_hash"].append(conv_hash)
            rows["n_user_turns"].append(n_u)
            rows["n_asst_turns"].append(n_a)
            rows["token_id"].append(int(tok_id))
            rows["token_str"].append(tok.decode([tok_id]))
            rows["position"].append(int(padded_pos))
            rows["unpadded_pos"].append(int(unpadded_pos))
            rows["role"].append(roles[unpadded_pos] or "unknown")
            rows["chars_before"].append(chars_b)
            rows["chars_after"].append(chars_a)
            rows["seq_len"].append(seq_len)
            rows["vec_norm"].append(float(np.linalg.norm(v_fp32)))
            rows["activation"].append(v_fp32.tolist())
            n_written += 1

    WINDOW = 50

    with Path(args.regen_jsonl).open() as fin:
        for line in fin:
            if n_written >= args.n:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = rec.get("messages") or []
            if not messages:
                continue
            try:
                ids, roles = format_and_tokenize_chat(messages, tok, args.max_len)
            except Exception as e:
                print(f"[extract] tok error: {e!r}", file=sys.stderr)
                continue
            if not ids:
                continue
            pos = pick_random_position(ids, rng=rng)
            if pos < 0:
                continue
            # chars_before / chars_after via decode of ids slices
            before_full = tok.decode(ids[:pos], skip_special_tokens=False)
            after_full = tok.decode(ids[pos+1:], skip_special_tokens=False)
            chars_b = before_full[-WINDOW:]
            chars_a = after_full[:WINDOW]

            pending.append((len(pending), rec.get("conv_hash"), ids, roles,
                            chars_b, chars_a, pos,
                            rec.get("n_user_turns", 0), rec.get("n_asst_turns", 0)))

            if len(pending) >= args.batch_size:
                flush_batch(pending)
                pending = []
                if len(rows["sample_idx"]) >= args.flush_every:
                    writer.write_table(pa.table(rows, schema=schema))
                    rows = {k: [] for k in schema.names}
                    elapsed = time.time() - t0
                    rate = n_written / max(elapsed, 1e-6)
                    eta = (args.n - n_written) / max(rate, 1e-6)
                    print(f"[extract] {n_written}/{args.n}  {rate:.1f}/s  "
                          f"eta={eta/60:.1f}min", flush=True)

        # Final batch
        if pending and n_written < args.n:
            flush_batch(pending)

    if rows["sample_idx"]:
        writer.write_table(pa.table(rows, schema=schema))
    writer.close()
    handle.remove()

    meta = {
        "regen_jsonl": str(args.regen_jsonl),
        "n_target": args.n,
        "n_written": n_written,
        "max_len": args.max_len,
        "base_model": args.base_model,
        "layer": args.layer,
        "seed": args.seed,
    }
    out_path.with_suffix(out_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2))
    print(f"[extract] done. wrote {n_written} rows in {(time.time()-t0)/60:.1f}min")


# ─── Phase 2: decode (no similarity computation) ─────────────────────────────

def cmd_decode(args):
    import httpx
    import orjson
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from nla_inference import EXPLANATION_RE, NLAClient, NLACritic

    av_client = NLAClient(args.av_checkpoint, sglang_url=args.sglang_url)
    critic = NLACritic(args.ar_checkpoint, device=args.ar_device)

    pf = pq.ParquetFile(args.activations)
    n_total = pf.metadata.num_rows
    act_col = "activation"
    d_act = pf.schema_arrow.field(act_col).type.list_size
    if d_act != av_client.cfg.d_model:
        sys.exit(f"d_model mismatch: {d_act} vs AV {av_client.cfg.d_model}")

    in_schema = pf.schema_arrow
    # Build output schema: all extract columns + explanation, raw_av_text, av_parsed, recon
    extra = pa.schema([
        pa.field("explanation", pa.string()),
        pa.field("raw_av_text", pa.string()),
        pa.field("av_parsed", pa.bool_()),
        pa.field("recon", pa.list_(pa.float32(), d_act)),
    ])
    out_schema = pa.schema(list(in_schema) + list(extra))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(args.out), out_schema, compression="zstd")

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "skip_special_tokens": False,
    }

    async def run():
        limits = httpx.Limits(max_connections=args.av_concurrency * 4,
                              max_keepalive_connections=args.av_concurrency)
        client = httpx.AsyncClient(timeout=httpx.Timeout(args.av_timeout),
                                    limits=limits)
        sem = asyncio.Semaphore(args.av_concurrency)

        async def call_av(vec_np: np.ndarray) -> str:
            embeds_np, _ = av_client._build_embeds(
                torch.as_tensor(vec_np, dtype=torch.float32), prompt_content=None
            )
            body = orjson.dumps(
                {"input_embeds": embeds_np, "sampling_params": sampling_params},
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
            last_exc = None
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
            raise RuntimeError(f"AV failed: {last_exc!r}")

        rows: dict[str, list] = {k: [] for k in out_schema.names}
        n_done = 0
        t0 = time.time()

        for batch in pf.iter_batches(batch_size=args.batch_size):
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
                # Copy all input cols
                for k in in_schema.names:
                    rows[k].append(d[k][i])
                rows["explanation"].append(expl)
                rows["raw_av_text"].append(raw)
                rows["av_parsed"].append(parsed)
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
                print(f"[decode] {n_done}/{n_total}  {rate:.1f}/s "
                      f"eta={eta/60:.1f}min", flush=True)

        if rows["sample_idx"]:
            writer.write_table(pa.table(rows, schema=out_schema))
        await client.aclose()

    asyncio.run(run())
    writer.close()
    print(f"[decode] done")


# ─── Smoke ───────────────────────────────────────────────────────────────────

def cmd_smoke(args):
    """CPU-only smoke: just tokenize + role-label a tiny fixture."""
    from transformers import AutoTokenizer
    # Use Qwen for smoke (no gating, downloads small)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great, thanks!"},
    ]
    # Qwen uses different chat template; this smoke is just to verify
    # the helper handles BOS-less templates gracefully.
    print("[smoke] OK — tokenizer loaded for", tok.name_or_path)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    pe = sp.add_parser("extract")
    pe.add_argument("--base-model", required=True)
    pe.add_argument("--layer", type=int, required=True)
    pe.add_argument("--regen-jsonl", required=True)
    pe.add_argument("--n", type=int, default=20_000)
    pe.add_argument("--max-len", type=int, default=2048)
    pe.add_argument("--batch-size", type=int, default=2)
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--device-map", default="auto")
    pe.add_argument("--out", required=True)
    pe.add_argument("--flush-every", type=int, default=256)
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
    pd_.add_argument("--out", required=True)
    pd_.set_defaults(func=cmd_decode)

    ps = sp.add_parser("smoke")
    ps.set_defaults(func=cmd_smoke)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
