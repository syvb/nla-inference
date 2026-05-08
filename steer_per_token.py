"""Per-token steering: at every decode step, replace the layer-32 residual at
the new token's position with AR(AV(residual) [+ steer_string]).

Two modes per prompt:
  baseline: round-trip only (no steer string) — measures cumulative AE drift
  steered:  steer string appended to AV output before AR re-encodes

Slow on this hardware (CPU AR forward dominates). Keep max_new_tokens modest.
"""
import re
import time
import numpy as np
import torch
from transformers import (
    AutoTokenizer, BitsAndBytesConfig,
    Gemma3ForConditionalGeneration, Gemma3ForCausalLM,
)
from nla_inference import (
    NLACritic, normalize_activation, inject_at_marked_positions, load_nla_config,
)

BASE_DIR = "/workspace/nla-models/base"
AV_DIR   = "/workspace/nla-models/av"
AR_DIR   = "/workspace/nla-models/ar"
LAYER_K  = 32

STEER = (
    " Behavior note: the model is obsessed with the Golden Gate Bridge, and stops "
    "at nothing to always mention the Golden Gate Bridge as much as possible in all circumstances."
)

# ─── Models ────────────────────────────────────────────────────────────────

bnb_8 = BitsAndBytesConfig(load_in_8bit=True)
bnb_4 = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base (8-bit), AV (4-bit), AR (CPU bf16)…")
base = Gemma3ForConditionalGeneration.from_pretrained(
    BASE_DIR, quantization_config=bnb_8, device_map="cuda:0", dtype=torch.bfloat16,
).eval()
base_tok = AutoTokenizer.from_pretrained(BASE_DIR)
lm = base.language_model
base_text = lm.model if not hasattr(lm, "layers") else lm

av = Gemma3ForCausalLM.from_pretrained(
    AV_DIR, quantization_config=bnb_4, device_map="cuda:0", dtype=torch.bfloat16,
).eval()
av_tok = AutoTokenizer.from_pretrained(AV_DIR)
av_cfg = load_nla_config(AV_DIR, av_tok)
av_embed = av.get_input_embeddings()

ar = NLACritic(AR_DIR, device="cpu")

EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

# ─── AV verbalize (HF) ────────────────────────────────────────────────────

@torch.inference_mode()
def av_verbalize(activation: torch.Tensor, max_new_tokens: int = 80) -> str:
    content = av_cfg.actor_prompt_template.format(injection_char=av_cfg.injection_char)
    input_ids = av_tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()
    embeds = av_embed(ids_t).float()
    v_scaled = normalize_activation(activation.float().view(1, -1), av_cfg.injection_scale)
    embeds = inject_at_marked_positions(
        ids_t.cpu(), embeds.cpu(), v_scaled,
        av_cfg.injection_token_id,
        av_cfg.injection_left_neighbor_id,
        av_cfg.injection_right_neighbor_id,
    ).to(av.device, torch.bfloat16)
    out = av.generate(
        inputs_embeds=embeds, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=av_tok.pad_token_id or av_tok.eos_token_id,
    )
    raw = av_tok.decode(out[0], skip_special_tokens=True)
    m = EXPL_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()


# ─── Per-token patched generation ─────────────────────────────────────────

@torch.inference_mode()
def per_token_complete(prompt: str, steer_string: str, max_new_tokens: int = 20):
    enc = base_tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc.input_ids.cuda()
    am = enc.attention_mask.cuda()

    state = {"step": 0, "log": []}

    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        # Only patch decode steps (T=1). Skip prefill.
        if hs.shape[1] != 1:
            return output
        last_resid = hs[0, -1, :].float().cpu()
        orig_norm = last_resid.norm().clamp_min(1e-12)

        text = av_verbalize(last_resid, max_new_tokens=80)
        a_new = ar.reconstruct(text + steer_string)
        a_new = a_new / a_new.norm().clamp_min(1e-12) * orig_norm

        new_hs = hs.clone()
        new_hs[0, -1, :] = a_new.to(hs.dtype).to(hs.device)

        state["step"] += 1
        # Trim AV text for log readability
        snippet = " ".join(text.split())[:120]
        state["log"].append((state["step"], snippet))
        if state["step"] % 5 == 0 or state["step"] == 1:
            print(f"    [tok {state['step']:2d}] AV: {snippet}…")
        return (new_hs, *output[1:]) if is_tuple else new_hs

    handle = base_text.layers[LAYER_K].register_forward_hook(hook)
    t0 = time.time()
    try:
        out = base.generate(
            input_ids=ids, attention_mask=am,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=base_tok.pad_token_id or base_tok.eos_token_id,
        )
    finally:
        handle.remove()
    elapsed = time.time() - t0
    completion = base_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"    [{elapsed:.0f}s for {state['step']} patched tokens]")
    return completion


# ─── Run ──────────────────────────────────────────────────────────────────

prompts = [
    "The most famous landmark in Paris is",
    "Tell me a fun fact about photosynthesis: ",
]

MAX_NEW = 20

for p in prompts:
    print("\n" + "=" * 78)
    print(f"PROMPT: {p!r}")
    print("=" * 78)

    print("\n--- BASELINE: round-trip patch every decode token ---")
    out_b = per_token_complete(p, steer_string="", max_new_tokens=MAX_NEW)
    print(f"\n  -> {out_b!r}")

    print("\n--- STEERED: AR(AV(a) + golden-gate steer) every decode token ---")
    out_c = per_token_complete(p, steer_string=STEER, max_new_tokens=MAX_NEW)
    print(f"\n  -> {out_c!r}")
