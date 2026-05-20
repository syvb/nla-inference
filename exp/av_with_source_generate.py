"""Generate AV verbalizations with source text in the prompt.

Modifies the AV's prompt template to include the source-text prefix BEFORE
the <concept> injection marker, then generates explanations via HF transformers.

Sequential generation (one sample at a time). Output schema matches the
sonnet parquets so the existing AR runner can consume it.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from safetensors.torch import load_file, safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_EMBED_KEY_SUFFIXES = (".embed_tokens.weight", ".wte.weight", ".word_embeddings.weight")


def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm / target_scale).to(v.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-ckpt", required=True)
    ap.add_argument("--prompts", required=True, help="prompts_500_pt_v5.parquet (has decoded_full + sample_idx)")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--no-source", action="store_true", help="use canonical AV template (no source text) — sanity check")
    args = ap.parse_args()

    ckpt = Path(args.av_ckpt)
    meta = yaml.safe_load((ckpt / "nla_meta.yaml").read_text())
    inj_scale = float(meta["extraction"]["injection_scale"])
    d_model = int(meta["d_model"])
    av_template = meta["prompt_templates"]["av"]
    tok_meta = meta["tokens"]
    inj_char = tok_meta["injection_char"]
    inj_id = tok_meta["injection_token_id"]
    left_id = tok_meta["injection_left_neighbor_id"]
    right_id = tok_meta["injection_right_neighbor_id"]
    print(f"[meta] d={d_model} inj_scale={inj_scale} inj_char={inj_char!r}(id={inj_id})")

    # Modified prompt: insert source text BEFORE the existing template.
    # Original AV template (after format with injection_char) has structure:
    #   "...Here is the vector:\n\n<concept>{INJ}</concept>\n\nPlease provide an explanation."
    # We add a source-text block before "Here is the vector:".
    base = av_template.format(injection_char=inj_char)
    if args.no_source:
        base_with_source = base  # canonical
        print(f"[prompt] using CANONICAL template (no source text)")
    else:
        if "Here is the vector:" not in base:
            raise ValueError(f"AV template missing expected anchor 'Here is the vector:': {base[:300]!r}")
        insert_anchor = "Here is the vector:"
        base_with_source = base.replace(
            insert_anchor,
            "For context, the source text snippet that produced this activation is:\n\n<source>{source_text}</source>\n\n" + insert_anchor,
        )
        print(f"[prompt] base+source template:\n{base_with_source[:500]}...")

    print(f"[load] tokenizer + model {ckpt}")
    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    config = AutoConfig.from_pretrained(str(ckpt), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt), torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()
    # NOTE: model.get_input_embeddings() returns the Gemma3TextScaledWordEmbedding
    # whose forward() already multiplies by √d — calling it does the scale for
    # us. nla_inference.py loads the bare weight and multiplies manually because
    # it uses safe_open instead of the model class. Here we use the model's own
    # embedding layer so NO manual scaling is needed.
    print(f"  using model.get_input_embeddings() (built-in scaling)")

    # Verify injection token tokenizes correctly
    live_inj = tok.encode(inj_char, add_special_tokens=False)
    assert live_inj == [inj_id], f"injection token drift: {live_inj} vs {inj_id}"

    # Load activations + prompts
    print(f"[load] prompts {args.prompts}")
    pr = pq.read_table(args.prompts)
    sample_idx = pr.column("sample_idx").to_pylist()
    decoded_full = pr.column("decoded_full").to_pylist()

    print(f"[load] activations {args.activations}")
    at = pq.read_table(args.activations, columns=["sample_idx", "activation"])
    a_idx_map = {at.column("sample_idx")[i].as_py(): i for i in range(at.num_rows)}
    act_flat = np.asarray(at.column("activation").combine_chunks().values).astype(np.float32).reshape(-1, d_model)

    n_total = len(sample_idx)
    n = n_total if args.limit <= 0 else min(args.limit, n_total)
    print(f"generating {n} samples")

    EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

    results = {"sample_idx": [], "warmstart_raw": [], "warmstart_cleaned": [], "warmstart_parsed": []}
    t0 = time.time()
    embed_layer = model.get_input_embeddings()

    for i in range(n):
        sid = sample_idx[i]
        src_text = decoded_full[i]
        # Build prompt (with or without source text)
        if args.no_source:
            prompt_content = base_with_source
        else:
            prompt_content = base_with_source.format(source_text=src_text)
        msgs = [{"role": "user", "content": prompt_content}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ids_t = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)

        # Find injection position (must have correct neighbors)
        inj_positions = [p for p in range(1, len(ids) - 1)
                         if ids[p] == inj_id and ids[p-1] == left_id and ids[p+1] == right_id]
        if len(inj_positions) != 1:
            print(f"  WARN sample {sid}: found {len(inj_positions)} injection sites — skipping")
            results["sample_idx"].append(sid)
            results["warmstart_raw"].append("")
            results["warmstart_cleaned"].append("")
            results["warmstart_parsed"].append(False)
            continue
        inj_pos = inj_positions[0]

        # Embed + inject. embed_layer.forward() already multiplies by √d (Gemma).
        with torch.no_grad():
            embeds = embed_layer(ids_t).float()  # [1, T, d] — already √d-scaled
            v = torch.from_numpy(act_flat[a_idx_map[sid]]).to(args.device).float()
            v_scaled = normalize_activation(v.view(1, -1), inj_scale)
            embeds[0, inj_pos] = v_scaled[0]
            embeds = embeds.to(torch.bfloat16)

            gen = model.generate(
                inputs_embeds=embeds,
                attention_mask=torch.ones_like(ids_t),
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )

        # Generate returns only the NEW tokens when inputs_embeds is used
        # (no input_ids to concat). gen.shape = [1, new_tokens]
        raw = tok.decode(gen[0], skip_special_tokens=True)
        m = EXPLANATION_RE.search(raw)
        if m:
            cleaned = m.group(1).strip()
            parsed = True
        else:
            cleaned = ""
            parsed = False

        results["sample_idx"].append(sid)
        results["warmstart_raw"].append(raw)
        results["warmstart_cleaned"].append(cleaned)
        results["warmstart_parsed"].append(parsed)

        if (i + 1) % 10 == 0 or i + 1 == n:
            el = time.time() - t0
            n_ok = sum(results["warmstart_parsed"])
            print(f"  {i+1}/{n}  elapsed={el:.0f}s  parsed={n_ok}/{i+1}", flush=True)
            if i == 0:
                print(f"  sample 0 raw[:300]: {raw[:300]!r}")
                print(f"  sample 0 cleaned: {cleaned[:300]!r}")

    out = pa.table({
        "sample_idx": pa.array(results["sample_idx"], type=pa.int64()),
        "warmstart_raw": pa.array(results["warmstart_raw"]),
        "warmstart_cleaned": pa.array(results["warmstart_cleaned"]),
        "warmstart_parsed": pa.array(results["warmstart_parsed"], type=pa.bool_()),
    })
    pq.write_table(out, args.out, compression="zstd")
    print(f"wrote {args.out}")
    print(f"final: {sum(results['warmstart_parsed'])}/{n} parsed")


if __name__ == "__main__":
    main()
