"""NLA image soft-token verbalizer — Gemma-3 AV on ZeroGPU.

Upload an image; Gemma-3-IT (the multimodal base model) turns it into 256 image
"soft tokens" and runs them through its residual stream; we then feed those
vectors to the matching NLA *activation verbaliser* (AV) and read back what it
says each one means.

This is firmly out-of-distribution: the AV was trained on text-derived residual
activations at one specific layer, never on vision soft tokens. The point is to
*look* at what falls out, not to expect fidelity.

Two halves, never on the GPU at the same time (so the 27B pair fits in 70 GB):
  1. EXTRACT  — base Gemma3ForConditionalGeneration encodes the image, we read
                hidden_states[depth] at the 256 image-token positions.
  2. VERBALISE — the AV (text-only Gemma3ForCausalLM) injects each [d_model]
                vector into its fixed prompt (the nla_inference.py recipe) and
                autoregresses an <explanation>.

Which pair loads is chosen by the NLA_MODEL env var ("12b" | "27b").
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import gradio as gr
import numpy as np
import spaces
import torch
import yaml
from huggingface_hub import hf_hub_download, login
from PIL import Image, ImageDraw, ImageFont
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
        "duration": 120,
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

    _STATE.update(
        base=base, processor=processor, av=av, av_tok=av_tok, av_meta=av_meta,
        av_prompt_ids=torch.tensor(ids, dtype=torch.long)[None],
        inj_pos=pos,
        n_layers=base.config.text_config.num_hidden_layers,
        image_token_id=base.config.image_token_index,
    )
    print(f"[load] done. layers={_STATE['n_layers']} d_model={av_meta['d_model']} "
          f"inj_pos={pos} inj_scale={av_meta['injection_scale']}")
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
    side = int(round(math.sqrt(vecs.shape[0])))
    return vecs, side


# ─── Token selection ────────────────────────────────────────────────────────────

def select_cells(side: int, mode: str, grid_n: int):
    """Returns list of (label, token_index) for the chosen verbalisation targets.
    Cells are sampled on the side×side spatial grid in raster order."""
    if mode == "Mean of all tokens":
        return [("mean", -1)]
    if mode.startswith("All"):
        return [(f"r{i // side},c{i % side}", i) for i in range(side * side)]
    # Downsampled grid
    n = max(1, min(int(grid_n), side))
    rows = sorted({int((i + 0.5) * side / n) for i in range(n)})
    cols = sorted({int((j + 0.5) * side / n) for j in range(n)})
    return [(f"r{r},c{c}", r * side + c) for r in rows for c in cols]


def annotate(image: Image.Image, side: int, cells, mean_mode: bool) -> Image.Image:
    """Overlay the side×side grid and highlight/number the selected cells."""
    img = image.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    cw, ch = W / side, H / side
    for k in range(side + 1):
        draw.line([(k * cw, 0), (k * cw, H)], fill=(255, 255, 255, 60), width=1)
        draw.line([(0, k * ch), (W, k * ch)], fill=(255, 255, 255, 60), width=1)
    if mean_mode:
        draw.rectangle([0, 0, W - 1, H - 1], outline=(255, 80, 80, 255), width=4)
        return img
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(10, int(ch * 0.35)))
    except Exception:
        font = ImageFont.load_default()
    for n, (_label, idx) in enumerate(cells):
        r, c = idx // side, idx % side
        x0, y0 = c * cw, r * ch
        draw.rectangle([x0, y0, x0 + cw, y0 + ch],
                       outline=(255, 200, 0, 255), width=2,
                       fill=(255, 200, 0, 40))
        draw.text((x0 + 3, y0 + 1), str(n), fill=(255, 80, 0, 255), font=font)
    return img


# ─── Main GPU pipeline ──────────────────────────────────────────────────────────

@spaces.GPU(duration=DURATION)
def run(image, mode, grid_n, depth, temperature, max_new_tokens,
        inj_scale, text_context, seed):
    if image is None:
        return None, "Upload an image first."
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))

    st = get_state()
    av_meta = st["av_meta"]
    inj_scale = float(inj_scale) if inj_scale and inj_scale > 0 else av_meta["injection_scale"]

    base, av = st["base"], st["av"]

    # Phase 1: encode image with the base model (base on GPU, AV on CPU).
    base.to("cuda")
    vecs, side = extract_soft_tokens(image, text_context, depth)
    base.to("cpu")
    torch.cuda.empty_cache()

    cells = select_cells(side, mode, grid_n)
    mean_mode = cells[0][1] == -1
    if mean_mode:
        targets = vecs.mean(0, keepdim=True)
        norms = [vecs.norm(dim=-1).mean().item()]
    else:
        idxs = torch.tensor([i for _, i in cells])
        targets = vecs[idxs]
        norms = vecs[idxs].norm(dim=-1).tolist()

    # Phase 2: verbalise with the AV (AV on GPU, base already off).
    av.to("cuda")
    texts = verbalize(targets, temperature=temperature,
                      max_new_tokens=int(max_new_tokens), inj_scale=inj_scale)
    av.to("cpu")
    torch.cuda.empty_cache()

    annotated = annotate(image, side, cells, mean_mode)

    header = (
        f"**{CFG['label']}** · {side}×{side} = {side * side} soft tokens · "
        f"extraction depth `hidden_states[{int(depth)}]` "
        f"(0 = raw soft tokens, {av_meta['layer_index'] + 1} = AV training layer) · "
        f"injection_scale `{inj_scale:g}`\n\n"
    )
    lines = []
    for n, ((label, _idx), text, nrm) in enumerate(zip(cells, texts, norms)):
        tag = "**mean of 256 soft tokens**" if mean_mode else f"**{n}** · `{label}`"
        lines.append(f"{tag} · ‖v‖={nrm:.0f}\n\n{_blockquote(text)}\n")
    return annotated, header + "\n".join(lines)


def _blockquote(text: str) -> str:
    """Prefix every line (incl. blank paragraph-break lines) with `>` so a
    multi-paragraph AV output stays inside one markdown blockquote."""
    return "\n".join(("> " + ln) if ln.strip() else ">"
                     for ln in text.strip().splitlines()) or "> (empty)"


# ─── UI ─────────────────────────────────────────────────────────────────────────

DESC = f"""# 🔍 NLA image soft-token verbaliser — {CFG['label']}

Upload an image. **{CFG['base']}** encodes it into image *soft tokens* and runs
them through its residual stream; the matching **NLA activation verbaliser**
([`{CFG['av']}`](https://huggingface.co/{CFG['av']})) then reads back what each
vector "means" in natural language.

⚠️ **Expect weirdness.** The AV was trained to verbalise *text*-derived residual
activations at one layer — never vision soft tokens. This is an out-of-distribution
probe for curiosity, not a captioning tool. CJK / nonsense output is an expected
failure mode, not a bug.
"""


def build():
    with gr.Blocks(title=f"NLA image verbaliser · {CFG['label']}") as demo:
        gr.Markdown(DESC)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Image")
                mode = gr.Radio(
                    ["Downsampled grid", "Mean of all tokens", "All tokens (slow)"],
                    value="Downsampled grid", label="What to verbalise",
                )
                grid_n = gr.Slider(1, 16, value=4, step=1,
                                   label="Grid size N (verbalises an N×N sample of cells)")
                depth = gr.Slider(
                    0, CFG["n_layers"],
                    value=CFG["av_layer"] + 1, step=1,
                    label="Extraction depth (hidden_states index)",
                )
                with gr.Accordion("Generation / advanced", open=False):
                    temperature = gr.Slider(0.0, 1.5, value=1.0, step=0.05,
                                            label="Temperature (0 = greedy)")
                    max_new_tokens = gr.Slider(32, 512, value=400, step=8,
                                               label="Max new tokens")
                    inj_scale = gr.Number(value=0, label="Injection-scale override "
                                          "(0 = use sidecar value)")
                    text_context = gr.Textbox(value="", label="Optional text shown "
                                              "with the image during encoding")
                    seed = gr.Number(value=0, label="Seed (-1 = random)")
                go = gr.Button("Verbalise", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(label="Selected soft-token cells", type="pil")
                out_md = gr.Markdown()
        go.click(
            run,
            [image, mode, grid_n, depth, temperature, max_new_tokens,
             inj_scale, text_context, seed],
            [out_img, out_md],
        )
    return demo


get_state()  # eager CPU load at import (downloads weights; no CUDA touched)
demo = build()
demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch()
