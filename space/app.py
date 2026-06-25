"""NLA image soft-token verbalizer — Gemma-3 AV on ZeroGPU.

Upload an image; Gemma-3-IT (the multimodal base model) turns it into 256 image
"soft tokens" and runs them through its residual stream; we then feed those
vectors to the matching NLA *activation verbaliser* (AV) and read back what it
says each one means.

This is firmly out-of-distribution: the AV was trained on text-derived residual
activations at one specific layer, never on vision soft tokens. The point is to
*look* at what falls out, not to expect fidelity.

Two halves:
  1. EXTRACT  — base Gemma3ForConditionalGeneration encodes the image, we read
                hidden_states[depth] at the 256 image-token positions.
  2. VERBALISE — the AV (text-only Gemma3ForCausalLM) injects each [d_model]
                vector into its fixed prompt (the nla_inference.py recipe) and
                autoregresses an <explanation>.

Which pair loads is chosen by the NLA_MODEL env var ("12b" | "27b").
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import gradio as gr
import spaces
import torch
import yaml
from huggingface_hub import hf_hub_download, login
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)

# ─── Model registry ───────────────────────────────────────────────────────────

MODELS = {
    "12b": {
        "base": "google/gemma-3-12b-it",
        "av": "kitft/nla-gemma3-12b-L32-av",
        "label": "Gemma-3-12B · AV layer 32",
        "n_layers": 48,      # base text layers -> hidden_states index range 0..48
        "av_layer": 32,      # AV training layer; default depth = av_layer + 1
        # ZeroGPU multiplies this by `duration_factor` (=2 here) for the actual
        # reservation, and anonymous (IP-quota) users have a low max. Real
        # runtime is ~12-15s incl. cold GPU materialization, so 30s (→60s
        # reserved) keeps logged-out users under the cap with headroom.
        "duration": 30,
    },
    "27b": {
        "base": "google/gemma-3-27b-it",
        "av": "kitft/nla-gemma3-27b-L41-av",
        "label": "Gemma-3-27B · AV layer 41",
        "n_layers": 62,
        "av_layer": 41,
        "duration": 300,
    },
}

KEY = os.environ.get("NLA_MODEL", "12b").lower()
assert KEY in MODELS, f"NLA_MODEL={KEY!r} not in {list(MODELS)}"
CFG = MODELS[KEY]
DURATION = CFG["duration"]

DTYPE = torch.bfloat16
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])


# ─── AV sidecar (the nla_meta.yaml contract) ───────────────────────────────────

def load_av_meta(av_repo: str, tokenizer) -> dict:
    """Parse nla_meta.yaml and assert the injection char survives the live
    tokenizer — the one cheap check that catches silent injection failure."""
    meta = yaml.safe_load(Path(hf_hub_download(av_repo, "nla_meta.yaml")).read_text())
    t = meta["tokens"]
    cfg = {
        "d_model": int(meta["d_model"]),
        "injection_char": t["injection_char"],
        "injection_token_id": int(t["injection_token_id"]),
        "left": int(t["injection_left_neighbor_id"]),
        "right": int(t["injection_right_neighbor_id"]),
        "injection_scale": float(meta["extraction"]["injection_scale"]),
        "template": meta["prompt_templates"]["av"],
        "layer_index": int(meta["extraction_layer_index"]),
    }
    live = tokenizer.encode(cfg["injection_char"], add_special_tokens=False)
    assert live == [cfg["injection_token_id"]], (
        f"tokenizer drift: {cfg['injection_char']!r} -> {live}, "
        f"sidecar says [{cfg['injection_token_id']}]"
    )
    return cfg


# ─── Lazy global load (CPU; moved to GPU per-phase inside the GPU fn) ───────────

_STATE: dict = {}


def get_state() -> dict:
    if _STATE:
        return _STATE
    base_repo, av_repo = CFG["base"], CFG["av"]
    print(f"[load] base={base_repo}  av={av_repo}")

    av_tok = AutoTokenizer.from_pretrained(av_repo)
    av_meta = load_av_meta(av_repo, av_tok)

    av = Gemma3ForCausalLM.from_pretrained(av_repo, dtype=DTYPE).eval()
    base = Gemma3ForConditionalGeneration.from_pretrained(
        base_repo, dtype=DTYPE
    ).eval()
    processor = AutoProcessor.from_pretrained(base_repo)

    assert av.config.hidden_size == av_meta["d_model"], "AV d_model mismatch"
    assert base.config.text_config.num_hidden_layers == CFG["n_layers"], (
        f"base layers {base.config.text_config.num_hidden_layers} != registry "
        f"{CFG['n_layers']}"
    )
    assert av_meta["layer_index"] == CFG["av_layer"], "AV layer mismatch"

    # Canonical AV prompt → token ids → injection position (computed once).
    content = av_meta["template"].format(injection_char=av_meta["injection_char"])
    ids = av_tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    ids = _to_id_list(ids)
    pos = _find_injection_pos(ids, av_meta)

    # ZeroGPU: place both models on the GPU once. `spaces` defers the actual
    # allocation to the first @spaces.GPU call and keeps the weights resident
    # across calls, so per-click inference pays no CPU<->GPU transfer. Both 12B
    # models (~48GB) fit together on the 70GB ZeroGPU H200, so no device-shuffle.
    av.to("cuda")
    base.to("cuda")

    mm = getattr(base.config, "mm_tokens_per_image", 256) or 256
    _STATE.update(
        base=base, processor=processor, av=av, av_tok=av_tok, av_meta=av_meta,
        av_prompt_ids=torch.tensor(ids, dtype=torch.long)[None],
        inj_pos=pos,
        n_layers=base.config.text_config.num_hidden_layers,
        image_token_id=base.config.image_token_index,
        side=int(round(math.sqrt(mm))),   # 256 soft tokens -> 16x16 spatial grid
    )
    print(f"[load] done. layers={_STATE['n_layers']} d_model={av_meta['d_model']} "
          f"inj_pos={pos} inj_scale={av_meta['injection_scale']} side={_STATE['side']}")
    return _STATE


def _to_id_list(ids) -> list[int]:
    """Coerce apply_chat_template output (list / BatchEncoding / tensor) to
    a flat list[int]."""
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids)
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return [int(t) for t in ids]


def _find_injection_pos(ids: list[int], meta: dict) -> int:
    """Locate the marker slot in our fully-controlled canonical prompt.

    Preferred: the marker token whose ±1 neighbours match the sidecar. But the
    recorded neighbour ids can drift with the tokenizer/chat-template version,
    and the prompt is built solely from the template (the marker char appears
    exactly once), so a single bare occurrence is safe to fall back on.
    """
    tid = meta["injection_token_id"]
    occ = [p for p, tok in enumerate(ids) if tok == tid]
    good = [
        p for p in occ
        if 0 < p < len(ids) - 1
        and ids[p - 1] == meta["left"] and ids[p + 1] == meta["right"]
    ]
    if len(good) == 1:
        return good[0]
    if len(occ) == 1:
        p = occ[0]
        nb = (ids[p - 1] if p > 0 else None, ids[p + 1] if p < len(ids) - 1 else None)
        print(f"[warn] marker neighbours {nb} != sidecar "
              f"({meta['left']},{meta['right']}); using sole occurrence at {p}")
        return p
    raise AssertionError(
        f"injection token {tid}: {len(occ)} occurrences "
        f"({len(good)} neighbour-matched) in {len(ids)}-token prompt"
    )


# ─── Verbalisation (AV side) ───────────────────────────────────────────────────

def _normalize(v: torch.Tensor, scale: float) -> torch.Tensor:
    n = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (v.float() / n) * scale


@torch.inference_mode()
def verbalize(vectors: torch.Tensor, *, temperature: float, max_new_tokens: int,
              inj_scale: float, chunk: int = 12) -> list[str]:
    """[N, d] residual vectors -> N explanation strings. Batched in chunks.

    Uses the AV's own (scaled) embedding for every non-injection position, then
    overwrites the marker slot with the L2-renormalised activation — exactly the
    nla_inference.py recipe, but local + batched instead of via SGLang.
    """
    st = get_state()
    av, tok = st["av"], st["av_tok"]
    dev = av.device
    pos = st["inj_pos"]
    prompt_ids = st["av_prompt_ids"].to(dev)              # [1, T]
    base_embeds = av.get_input_embeddings()(prompt_ids)   # [1, T, d] (×√d applied)
    T = prompt_ids.shape[1]

    vectors = vectors.to(dev)
    out: list[str] = []
    for i in range(0, vectors.shape[0], chunk):
        vchunk = vectors[i:i + chunk]
        b = vchunk.shape[0]
        emb = base_embeds.expand(b, T, -1).clone()
        emb[:, pos, :] = _normalize(vchunk, inj_scale).to(emb.dtype)
        attn = torch.ones(b, T, dtype=torch.long, device=dev)

        gen_kw = dict(max_new_tokens=max_new_tokens, attention_mask=attn)
        if temperature and temperature > 0:
            gen_kw.update(do_sample=True, temperature=float(temperature), top_p=0.95)
        else:
            gen_kw.update(do_sample=False)
        gen = av.generate(inputs_embeds=emb, **gen_kw)
        for text in tok.batch_decode(gen, skip_special_tokens=True):
            m = EXPLANATION_RE.search(text)
            # Fallback (no closing tag — truncated gen): drop the dangling
            # opening <explanation> so the raw text reads cleanly.
            out.append(m.group(1).strip() if m
                       else text.split("<explanation>")[-1].strip())
    return out


# ─── Extraction (base multimodal side) ─────────────────────────────────────────

@torch.inference_mode()
def extract_soft_tokens(image: Image.Image, text_context: str, depth: int):
    """Returns (vectors[256, d] fp32 cpu, grid_side). Reads hidden_states[depth]
    at the image-token positions. depth=0 == the raw soft tokens (the projector
    output that gets injected into the LM); depth=L+1 == output of block L."""
    st = get_state()
    base, proc = st["base"], st["processor"]
    dev = base.device

    content = [{"type": "image", "image": image}]
    if text_context.strip():
        content.append({"type": "text", "text": text_context.strip()})
    inputs = proc.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", do_pan_and_scan=False,
    )
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)

    out = base(**inputs, output_hidden_states=True)
    hs = out.hidden_states                       # tuple len n_layers+1
    depth = max(0, min(int(depth), len(hs) - 1))
    h = hs[depth][0]                             # [T, d]

    img_mask = inputs["input_ids"][0] == st["image_token_id"]
    idxs = img_mask.nonzero(as_tuple=True)[0]
    vecs = h[idxs].float().cpu()                 # [n_img, d]
    # The UI grid is fixed at st["side"]×st["side"] and the click->index mapping
    # assumes exactly that many tokens. do_pan_and_scan=False pins gemma-3 to
    # 256; assert so any future drift fails loud instead of mis-indexing cells.
    assert vecs.shape[0] == st["side"] ** 2, (
        f"got {vecs.shape[0]} image tokens, expected {st['side'] ** 2} "
        f"(pan&scan or multi-image?)"
    )
    return vecs, st["side"]


def _blockquote(text: str) -> str:
    """Prefix every line (incl. blank paragraph-break lines) with `>` so a
    multi-paragraph AV output stays inside one markdown blockquote."""
    return "\n".join(("> " + ln) if ln.strip() else ">"
                     for ln in text.strip().splitlines()) or "> (empty)"


# ─── Grid drawing (CPU only — no model needed) ─────────────────────────────────

def draw_grid(image: Image.Image, side: int,
              highlight: tuple[int, int] | None = None,
              border: bool = False) -> Image.Image:
    """Overlay a side×side grid; optionally highlight one (row, col) cell, or
    draw a full border (used for the whole-image mean)."""
    img = image.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    cw, ch = W / side, H / side
    for k in range(side + 1):
        draw.line([(k * cw, 0), (k * cw, H)], fill=(255, 255, 255, 70), width=1)
        draw.line([(0, k * ch), (W, k * ch)], fill=(255, 255, 255, 70), width=1)
    if highlight is not None:
        r, c = highlight
        x0, y0 = c * cw, r * ch
        draw.rectangle([x0, y0, x0 + cw, y0 + ch],
                       outline=(255, 60, 0, 255), width=3, fill=(255, 160, 0, 80))
    if border:
        draw.rectangle([1, 1, W - 2, H - 2], outline=(255, 60, 0, 255), width=4)
    return img


def _cell_from_xy(xy, image: Image.Image, side: int) -> tuple[int, int, int]:
    """Click pixel (x, y) -> (token_index, row, col) on the side×side grid."""
    x, y = float(xy[0]), float(xy[1])
    W, H = image.size
    c = min(side - 1, max(0, int(x / (W / side))))
    r = min(side - 1, max(0, int(y / (H / side))))
    return r * side + c, r, c


def _report(depth, idx, row, col, nrm, text, cached, scale) -> str:
    st = get_state()
    av_meta = st["av_meta"]
    where = (f"**mean of all {st['side'] ** 2} soft tokens**" if idx < 0
             else f"**cell {idx}** · row {row}, col {col}")
    badge = "↩︎ from cache" if cached else "✨ freshly generated"
    head = (f"{where} · ‖v‖={nrm:.0f} · {badge}\n\n"
            f"<sub>`{CFG['label']}` · depth `hidden_states[{int(depth)}]` "
            f"(0 = raw soft tokens, {av_meta['layer_index'] + 1} = AV training "
            f"layer) · injection_scale `{scale:g}`</sub>\n\n")
    return head + _blockquote(text)


def _resolve_scale(inj_scale) -> float:
    av_meta = get_state()["av_meta"]
    return float(inj_scale) if inj_scale and inj_scale > 0 else av_meta["injection_scale"]


# Generation params the precache.json was built with (kept here as the fallback
# for entries that predate the per-entry fields — the bundled JSON now records
# them explicitly so it never needs regenerating to pick up this change).
PRECACHE_GEN = {"temperature": 1.0, "max_new_tokens": 400, "inj_scale": 0}


def _gen_sig(temperature, max_new_tokens, inj_scale) -> str:
    """Canonical signature of the generation settings that affect the output
    text. Any change here must invalidate cached cells."""
    return (f"t{float(temperature):g}_m{int(max_new_tokens)}"
            f"_s{float(inj_scale):g}")


def _cache_sig(image: Image.Image, depth, temperature, max_new_tokens,
               inj_scale) -> str:
    """Signature of everything a cached explanation depends on: image bytes,
    extraction depth, and the generation settings. Stored under '__sig__'; a
    mismatch means the cached cells are stale (wrong image/depth/settings) and
    must be discarded. Changing temperature / max-tokens / injection-scale now
    invalidates the cache instead of silently serving old text."""
    h = hashlib.md5(image.tobytes()).hexdigest()[:12]
    return f"{h}:{int(depth)}:{_gen_sig(temperature, max_new_tokens, inj_scale)}"


def _fresh_cache(cache, image, depth, temperature, max_new_tokens,
                 inj_scale) -> dict:
    """Return cache if it matches the current signature, else a fresh cache
    carrying the current signature."""
    cache = dict(cache or {})
    sig = _cache_sig(image, depth, temperature, max_new_tokens, inj_scale)
    if cache.get("__sig__") != sig:
        return {"__sig__": sig}
    return cache


def _seed_cache(image, depth, temperature, max_new_tokens, inj_scale):
    """Initial cache for a freshly shown image at the given settings, returned
    as (cache, precached?). If the image is a bundled example whose precache was
    generated at exactly these settings (depth + generation params), inject the
    precomputed cells; otherwise an empty cache carrying the signature."""
    sig = _cache_sig(image, depth, temperature, max_new_tokens, inj_scale)
    h = sig.split(":", 1)[0]
    pc = PRECACHE.get(h)
    if pc is not None:
        pc_sig = (f"{h}:{int(pc['depth'])}:" + _gen_sig(
            pc.get("temperature", PRECACHE_GEN["temperature"]),
            pc.get("max_new_tokens", PRECACHE_GEN["max_new_tokens"]),
            pc.get("inj_scale", PRECACHE_GEN["inj_scale"])))
        if pc_sig == sig:
            cache = dict(pc["cells"])
            cache["__sig__"] = sig
            return cache, True
    return {"__sig__": sig}, False


# ─── GPU work: encode (once, cached in State) + verbalise one target ───────────

@spaces.GPU(duration=DURATION)
def gpu_encode_verbalize(image, depth, vstate, idx, temperature,
                         max_new_tokens, inj_scale):
    """Returns ((depth, vecs_np), text, norm, scale). `vstate` is the prior
    (encoded_depth, vecs_np) or None. The 256 vectors are reused across cells
    ONLY if they were encoded at the current depth — making depth authoritative
    regardless of event ordering (guards the depth-change-then-click race)."""
    get_state()  # ensure loaded; both models are already GPU-resident
    scale = _resolve_scale(inj_scale)
    depth = int(depth)

    if vstate is not None and vstate[0] == depth:
        vecs_np = vstate[1]
    else:
        vecs, _side = extract_soft_tokens(image, "", depth)
        vecs_np = vecs.numpy()

    vecs_t = torch.from_numpy(vecs_np)
    if idx is not None and idx >= 0:
        target = vecs_t[idx:idx + 1]
        nrm = float(vecs_t[idx].norm())
    else:
        target = vecs_t.mean(0, keepdim=True)
        nrm = float(vecs_t.norm(dim=-1).mean())

    text = verbalize(target, temperature=float(temperature),
                     max_new_tokens=int(max_new_tokens), inj_scale=scale)[0]
    return (depth, vecs_np), text, nrm, scale


@spaces.GPU(duration=DURATION)
def gpu_precache(image, depth, start, count, temperature, max_new_tokens, inj_scale):
    """Offline helper (hidden api_name='precache') used to build
    examples/precache.json: verbalise cells [start, start+count) — or the mean
    when start<0 — and return {hash, depth, cells}. `hash` is md5 of the
    gradio-loaded image bytes, the exact key on_new_image looks up at runtime."""
    get_state()
    scale = _resolve_scale(inj_scale)
    depth = int(depth)
    vecs, _ = extract_soft_tokens(image, "", depth)
    h = hashlib.md5(image.tobytes()).hexdigest()[:12]
    start, count = int(start), int(count)
    if start < 0:
        text = verbalize(vecs.mean(0, keepdim=True), temperature=float(temperature),
                         max_new_tokens=int(max_new_tokens), inj_scale=scale)[0]
        cells = {"mean": {"text": text,
                          "nrm": float(vecs.norm(dim=-1).mean()), "scale": scale}}
    else:
        e = min(start + count, vecs.shape[0])
        texts = verbalize(vecs[start:e], temperature=float(temperature),
                          max_new_tokens=int(max_new_tokens), inj_scale=scale)
        norms = vecs[start:e].norm(dim=-1).tolist()
        cells = {str(start + i): {"text": texts[i], "nrm": norms[i], "scale": scale}
                 for i in range(e - start)}
    return {"hash": h, "depth": depth, "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens), "inj_scale": float(inj_scale),
            "cells": cells}


# ─── CPU event handlers (GPU is only touched on a cache miss) ──────────────────

def on_new_image(image, depth, temperature, max_new_tokens, inj_scale):
    """Upload/clear/example -> draw the grid instantly (no GPU) and (re)seed the
    cache. A bundled example whose precache matches the current settings gets its
    precomputed cells injected so every click is instant with zero GPU."""
    if image is None:
        return None, "Upload an image to begin.", None, {}
    side = get_state()["side"]
    grid = draw_grid(image, side)
    cache, precached = _seed_cache(image, depth, temperature, max_new_tokens, inj_scale)
    if precached:
        msg = (f"### Example image — all {side * side} cells precached ✨\n"
               f"Click any cell (or **mean**) for an instant explanation. "
               f"Changing depth or generation settings regenerates on demand.")
    else:
        msg = (f"### Click any cell to verbalise its soft token\n"
               f"{side}×{side} = {side * side} image soft tokens. "
               f"The first click encodes the image (~10–15s); later clicks reuse "
               f"that encoding, and revisiting a cell is instant (cached).")
    return grid, msg, None, cache       # grid, md, vecs_state(reset), cache


def on_depth_change(image, depth, temperature, max_new_tokens, inj_scale):
    """Depth changes the residual layer we read -> invalidate the encoding and
    re-seed the cache (re-injecting the precache if settings line back up)."""
    if image is None:
        return gr.update(), None, {}, gr.update()
    side = get_state()["side"]
    cache, precached = _seed_cache(image, depth, temperature, max_new_tokens, inj_scale)
    note = ("all cells precached ✨" if precached
            else "cache cleared — clicks regenerate at the new settings")
    return (draw_grid(image, side), None, cache,
            f"Depth set to `hidden_states[{int(depth)}]` — {note}.")


def on_settings_change(image, depth, temperature, max_new_tokens, inj_scale):
    """A generation setting changed -> re-seed the text cache (so stale cells are
    dropped and the precache re-injects when settings line back up). Leaves the
    image encoding (vecs_state) and the grid untouched."""
    if image is None:
        return gr.update(), gr.update()
    side = get_state()["side"]
    cache, precached = _seed_cache(image, depth, temperature, max_new_tokens, inj_scale)
    if precached:
        msg = (f"### Example image — all {side * side} cells precached ✨\n"
               f"These settings match the precache, so every click is instant.")
    else:
        msg = ("### Settings changed\nClicks now generate at the new settings "
               "(cached cells from the old settings were dropped).")
    return msg, cache


def on_click(evt: gr.SelectData, image, depth, vstate, cache,
             temperature, max_new_tokens, inj_scale):
    if image is None or evt is None or evt.index is None:
        return gr.update(), gr.update(), vstate, cache or {}
    side = get_state()["side"]
    idx, row, col = _cell_from_xy(evt.index, image, side)
    # drop stale entries (wrong image / depth / generation settings)
    cache = _fresh_cache(cache, image, depth, temperature, max_new_tokens, inj_scale)
    key = str(idx)
    hit = key in cache
    if hit:
        entry = cache[key]
        text, nrm, scale = entry["text"], entry["nrm"], entry["scale"]
    else:
        vstate, text, nrm, scale = gpu_encode_verbalize(
            image, depth, vstate, idx, temperature, max_new_tokens, inj_scale)
        cache[key] = {"text": text, "nrm": nrm, "scale": scale}
    grid = draw_grid(image, side, highlight=(row, col))
    return grid, _report(depth, idx, row, col, nrm, text, hit, scale), vstate, cache


def on_mean(image, depth, vstate, cache, temperature, max_new_tokens, inj_scale):
    if image is None:
        return gr.update(), "Upload an image first.", vstate, cache or {}
    side = get_state()["side"]
    cache = _fresh_cache(cache, image, depth, temperature, max_new_tokens, inj_scale)
    hit = "mean" in cache
    if hit:
        entry = cache["mean"]
        text, nrm, scale = entry["text"], entry["nrm"], entry["scale"]
    else:
        vstate, text, nrm, scale = gpu_encode_verbalize(
            image, depth, vstate, -1, temperature, max_new_tokens, inj_scale)
        cache["mean"] = {"text": text, "nrm": nrm, "scale": scale}
    grid = draw_grid(image, side, border=True)
    return grid, _report(depth, -1, 0, 0, nrm, text, hit, scale), vstate, cache


# ─── UI ─────────────────────────────────────────────────────────────────────────

DESC = f"""# 🔍 NLA image soft-token verbaliser — {CFG['label']}

This encodes an uploaded image into soft tokens and then uses the Gemma NLA
to verbalize what the model is thinking about when it looks at those image soft
tokens. You can click on grid cells to verbalize the soft token at that grid cell.

This is OOD for the NLA as it wasn't trained on image inputs, but it does seem
to have generalized somewhat well.
"""

# Bundled sample images (downscaled to the 896px the vision encoder uses,
# metadata stripped). Paths are relative to the app root on the Space.
EXAMPLES = [
    ("examples/subway.jpg", "Subway"),
    ("examples/breaker_panel.jpg", "Breaker panel"),
]


def _load_precache() -> dict:
    """{img_hash: {"depth": int, "cells": {idx|"mean": {text,nrm,scale}}}} for the
    bundled examples, generated offline via the gpu_precache endpoint. Empty if
    absent (examples then fall back to on-demand generation)."""
    p = Path(__file__).parent / "examples" / "precache.json"
    if not p.exists():
        return {}
    try:
        pc = json.loads(p.read_text())
        print(f"[precache] loaded {len(pc)} example(s): "
              f"{ {k: len(v['cells']) for k, v in pc.items()} }")
        return pc
    except Exception as e:  # pragma: no cover
        print(f"[warn] precache load failed: {e}")
        return {}


PRECACHE = _load_precache()


def build():
    with gr.Blocks(title=f"NLA image verbaliser · {CFG['label']}") as demo:
        gr.Markdown(DESC)
        # Server-side: the 256 encoded vectors for the current image+depth.
        vecs_state = gr.State(None)
        # Browser-side (localStorage): {token_idx -> {text, norm}} explanation
        # cache. Reset whenever a new image is uploaded or depth changes.
        # Fall back to server-side State if BrowserState is unavailable.
        try:
            cache_state = gr.BrowserState({}, storage_key=f"nla_av_cache_{KEY}")
        except (AttributeError, TypeError) as e:
            print(f"[warn] BrowserState unavailable ({e}); using server State")
            cache_state = gr.State({})
        with gr.Row():
            with gr.Column(scale=1):
                uploader = gr.Image(type="pil", sources=["upload", "clipboard"],
                                    label="Upload / change image")
                gr.Examples(
                    examples=[[p] for p, _ in EXAMPLES],
                    example_labels=[lbl for _, lbl in EXAMPLES],
                    inputs=[uploader],
                    label="Or pick a provided image",
                )
                mean_btn = gr.Button("Verbalise mean of all soft tokens")
                with gr.Accordion("Generation / advanced", open=False):
                    temperature = gr.Slider(0.0, 1.5, value=1.0, step=0.05,
                                            label="Temperature (0 = greedy)")
                    max_new_tokens = gr.Slider(32, 512, value=400, step=8,
                                               label="Max new tokens")
                    inj_scale = gr.Number(value=0, label="Injection-scale override "
                                          "(0 = use sidecar value)")
                    depth = gr.Slider(
                        0, CFG["n_layers"], value=CFG["av_layer"] + 1, step=1,
                        label="Extraction depth (hidden_states index)",
                    )
            with gr.Column(scale=1):
                grid_img = gr.Image(type="pil", interactive=False,
                                    label="Click a soft-token cell")
                out_md = gr.Markdown("Upload an image to begin.")

        # Hidden endpoint used offline to generate examples/precache.json
        # (api_name='precache'); invisible, harmless to leave deployed.
        pc_start = gr.Number(0, visible=False)
        pc_count = gr.Number(16, visible=False)
        pc_out = gr.JSON(visible=False)
        pc_btn = gr.Button(visible=False)
        pc_btn.click(
            gpu_precache,
            [uploader, depth, pc_start, pc_count, temperature, max_new_tokens, inj_scale],
            [pc_out], api_name="precache",
        )

        gen_inputs = [temperature, max_new_tokens, inj_scale]
        seed_inputs = [uploader, depth, *gen_inputs]
        uploader.change(on_new_image, seed_inputs,
                        [grid_img, out_md, vecs_state, cache_state])
        depth.change(on_depth_change, seed_inputs,
                     [grid_img, vecs_state, cache_state, out_md])
        # Changing a generation setting re-seeds only the text cache + message
        # (the image encoding in vecs_state doesn't depend on these, so it's
        # preserved). Sliders fire on release to avoid thrashing during a drag.
        temperature.release(on_settings_change, seed_inputs, [out_md, cache_state])
        max_new_tokens.release(on_settings_change, seed_inputs, [out_md, cache_state])
        inj_scale.change(on_settings_change, seed_inputs, [out_md, cache_state])
        grid_img.select(on_click,
                        [uploader, depth, vecs_state, cache_state, *gen_inputs],
                        [grid_img, out_md, vecs_state, cache_state])
        mean_btn.click(on_mean,
                       [uploader, depth, vecs_state, cache_state, *gen_inputs],
                       [grid_img, out_md, vecs_state, cache_state])
    return demo


get_state()  # eager CPU load at import (downloads weights; no CUDA touched)
demo = build()
demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch()
