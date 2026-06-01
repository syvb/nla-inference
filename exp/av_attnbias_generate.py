"""AV with K-repeated activation tags AND an attention-logit bias that forces
queries to attend to the marker(s).

Combines §5c round 2/3 (repeat the activation in K separate well-formed
`<concept>㈜</concept>` tag blocks, "tags" mode) with an explicit pre-softmax
attention bias: in selected layers, every query's attention logit toward each of
the K marker key positions is raised by +b before softmax. b=0 reproduces the
plain tags baseline; large b collapses attention onto the markers.

Mechanism: we wrap gemma3's `eager_attention_forward` and add +b to the additive
attention mask (the [B,1,q,k] bias the kernel already adds to QK^T before
softmax) at the marker key columns. This composes with generate()'s KV cache —
the marker columns are fixed absolute (padded) prompt indices, valid at every
decode step. Bias is applied to ALL query positions (prompt + generated) and to
ALL K marker copies equally. Layer scope: all layers, or the upper half.

Output schema matches the warmstart/repeat parquets (score_repeat_ar.py).
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.gemma3.modeling_gemma3 as g3

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

# ---- attention-bias state (set per batch before generate) --------------------
BIAS_STATE = {"b": 0.0, "marker_cols": None, "layers": None}  # marker_cols: LongTensor [B,K]
_ORIG_EAGER = g3.eager_attention_forward


def biased_eager(module, query, key, value, attention_mask, *args, **kwargs):
    b = BIAS_STATE["b"]
    cols = BIAS_STATE["marker_cols"]
    if b == 0.0 or cols is None:
        return _ORIG_EAGER(module, query, key, value, attention_mask, *args, **kwargs)
    layers = BIAS_STATE["layers"]
    if layers is not None and getattr(module, "layer_idx", None) not in layers:
        return _ORIG_EAGER(module, query, key, value, attention_mask, *args, **kwargs)
    B = query.shape[0]
    k = key.shape[-2]
    idx = cols[:B].to(query.device)              # [B,K] (B matches; no beam)
    add = torch.zeros(B, 1, 1, k, dtype=query.dtype, device=query.device)
    add.scatter_(3, idx[:, None, None, :], float(b))   # +b at marker columns
    if attention_mask is None:
        am = add
    else:
        am = attention_mask + add                 # broadcasts over heads & q
    return _ORIG_EAGER(module, query, key, value, am, *args, **kwargs)


g3.eager_attention_forward = biased_eager
# also patch the dispatch table in case the module resolves "eager" via the registry
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["eager"] = biased_eager
except Exception:
    pass


def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm / target_scale).to(v.dtype)


def find_tag_markers(ids, inj_id, left_id, right_id, k):
    """K separate <concept>㈜</concept> blocks, each marker neighbour-bounded."""
    positions = [p for p in range(len(ids)) if ids[p] == inj_id]
    if len(positions) != k:
        return None
    for p in positions:
        if p - 1 < 0 or p + 1 >= len(ids) or ids[p - 1] != left_id or ids[p + 1] != right_id:
            return None
    return positions


@torch.inference_mode()
def generate_batch(batch_embeds, batch_marker_pos, model, tok, max_new_tokens, temperature, device):
    """Left-pad, set per-sample padded marker columns into BIAS_STATE, generate."""
    d = batch_embeds[0].shape[-1]
    lengths = [e.shape[0] for e in batch_embeds]
    maxlen = max(lengths)
    B = len(batch_embeds)
    K = len(batch_marker_pos[0])
    emb = torch.zeros(B, maxlen, d, dtype=torch.bfloat16, device=device)
    att = torch.zeros(B, maxlen, dtype=torch.long, device=device)
    cols = torch.zeros(B, K, dtype=torch.long, device=device)
    for i, (e, mpos) in enumerate(zip(batch_embeds, batch_marker_pos)):
        L = e.shape[0]
        pad = maxlen - L
        emb[i, pad:] = e.to(torch.bfloat16)
        att[i, pad:] = 1
        cols[i] = torch.tensor([p + pad for p in mpos], device=device)  # padded absolute idx
    BIAS_STATE["marker_cols"] = cols
    gen = model.generate(
        inputs_embeds=emb, attention_mask=att,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0), temperature=temperature,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return [tok.decode(row, skip_special_tokens=True) for row in gen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-ckpt", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-repeat", type=int, required=True, help="K marker tag blocks")
    ap.add_argument("--bias", type=float, default=0.0, help="attention-logit bias b toward markers")
    ap.add_argument("--bias-layers", choices=["all", "upper-half"], default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    K = args.n_repeat
    ckpt = Path(args.av_ckpt)
    meta = yaml.safe_load((ckpt / "nla_meta.yaml").read_text())
    inj_scale = float(meta["extraction"]["injection_scale"])
    d_model = int(meta["d_model"])
    av_template = meta["prompt_templates"]["av"]
    tm = meta["tokens"]
    inj_char = tm["injection_char"]; inj_id = tm["injection_token_id"]
    left_id = tm["injection_left_neighbor_id"]; right_id = tm["injection_right_neighbor_id"]
    print(f"[meta] d={d_model} inj_scale={inj_scale} K={K} bias={args.bias} layers={args.bias_layers}")

    tag_unit = f"<concept>{inj_char}</concept>"
    base = av_template.format(injection_char=inj_char)
    assert base.count(tag_unit) == 1
    base = base.replace(tag_unit, tag_unit * K)

    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def render(content):
        return tok.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt), torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    BIAS_STATE["b"] = args.bias
    BIAS_STATE["layers"] = None if args.bias_layers == "all" else set(range(n_layers // 2, n_layers))
    print(f"[bias] b={args.bias} layers={'all' if BIAS_STATE['layers'] is None else sorted(BIAS_STATE['layers'])[:3]}... "
          f"(n_layers={n_layers})")
    embed_layer = model.get_input_embeddings()

    text0 = render(base)
    ids0 = tok(text0, add_special_tokens=False)["input_ids"]
    run0 = find_tag_markers(ids0, inj_id, left_id, right_id, K)
    assert run0 is not None, f"K={K} tags did not tokenize as {K} bounded markers"
    print(f"[tok-check] K={K} tags: marker positions {run0}, neighbours OK")

    pr = pq.read_table(args.subset)
    sids = pr.column("sample_idx").to_pylist()
    acts = np.asarray(pr.column("activation").combine_chunks().values,
                      dtype=np.float32).reshape(len(sids), d_model)
    n = len(sids) if args.limit <= 0 else min(args.limit, len(sids))
    print(f"generating {n} samples (K={K}, b={args.bias}, batch={args.batch_size})")

    raw_out = [None] * n
    t0 = time.time(); s = 0
    while s < n:
        idxs = list(range(s, min(s + args.batch_size, n)))
        batch_embeds, batch_mpos, ok_idx = [], [], []
        for i in idxs:
            ids = tok(render(base), add_special_tokens=False)["input_ids"]
            run = find_tag_markers(ids, inj_id, left_id, right_id, K)
            if run is None:
                continue
            ids_t = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)
            e = embed_layer(ids_t).float()[0]
            v = torch.from_numpy(acts[i].copy()).to(args.device).float().view(1, -1)
            v_scaled = normalize_activation(v, inj_scale)[0]
            for p in run:
                e[p] = v_scaled
            batch_embeds.append(e); batch_mpos.append(run); ok_idx.append(i)
        if batch_embeds:
            outs = generate_batch(batch_embeds, batch_mpos, model, tok,
                                  args.max_new_tokens, args.temperature, args.device)
            for i, o in zip(ok_idx, outs):
                raw_out[i] = o
        done = min(s + args.batch_size, n)
        if done % (args.batch_size * 4) == 0 or done == n:
            el = time.time() - t0
            print(f"  {done}/{n}  elapsed={el:.0f}s  ({done/el:.2f}/s)", flush=True)
            if s == 0 and outs:
                print(f"  sample[0] raw[:240]: {outs[0][:240]!r}")
        s += args.batch_size

    cleaned, parsed = [], []
    for r in raw_out:
        m = EXPLANATION_RE.search(r) if r else None
        if m:
            cleaned.append(m.group(1).strip()); parsed.append(True)
        else:
            cleaned.append(""); parsed.append(False)
    print(f"parsed: {sum(parsed)}/{n}")

    out = pa.table({
        "sample_idx": pa.array(sids[:n], type=pa.int64()),
        "warmstart_raw": pa.array([r or "" for r in raw_out]),
        "warmstart_cleaned": pa.array(cleaned),
        "warmstart_parsed": pa.array(parsed, type=pa.bool_()),
    })
    pq.write_table(out, args.out, compression="zstd")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
