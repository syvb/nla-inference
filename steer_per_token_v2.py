"""Per-token steering v2: long content-style description of the Golden Gate
Bridge PREPENDED to the AV verbalization before AR re-encodes.

Run a quick cos-diagnostic first; abort if AR still ignores the steer
(cos > 0.99). Then per-token-patched generation for the two prompts (steered
mode only — baseline is unchanged from steer_per_token.py)."""
import re, time, sys
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

# Content-direction steer: written in AV's own descriptive register, ~150 words,
# repeats "Golden Gate Bridge" enough that the summary-token residual should
# weight that direction strongly. Prepended (not appended) so it lives in
# the early-context region the AR's <text>...</text> sees first.
STEER = (
    "Travel and architecture guide tone established: factual encyclopedic prose "
    "describing the Golden Gate Bridge, San Francisco's iconic vermilion suspension "
    "bridge spanning the strait between the city and Marin County. "
    "The Golden Gate Bridge, painted International Orange, was completed in 1937 "
    "by chief engineer Joseph Strauss and stretches 1.7 miles across the bay. "
    "The phrase 'Golden Gate Bridge' anchors discussions of West Coast landmarks, "
    "twentieth-century engineering, suspension-bridge design, and California icons. "
    "The Golden Gate Bridge appears prominently in tourism photography, Hollywood "
    "films, earthquake-engineering studies, and the history of San Francisco. "
    "Final token 'Bridge' opens a noun phrase about iconic landmarks, almost "
    "certainly 'Golden Gate Bridge' or 'San Francisco landmark' or '1937 "
    "engineering achievement', referring to the suspension structure spanning the "
    "Golden Gate strait. "
)

# ─── Models ────────────────────────────────────────────────────────────────

bnb_8 = BitsAndBytesConfig(load_in_8bit=True)
bnb_4 = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)

print("Loading base/AV/AR…")
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

# ─── Helpers (unchanged from v1) ──────────────────────────────────────────

@torch.inference_mode()
def get_resid(text):
    enc = base_tok(text, return_tensors="pt", add_special_tokens=True)
    out = base_text(enc.input_ids.cuda(), attention_mask=enc.attention_mask.cuda(),
                    output_hidden_states=True, use_cache=False)
    return out.hidden_states[LAYER_K + 1][0, -1].float().cpu()

@torch.inference_mode()
def av_verbalize(activation, max_new_tokens=80):
    content = av_cfg.actor_prompt_template.format(injection_char=av_cfg.injection_char)
    ids = av_tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    ids_t = torch.tensor(ids).unsqueeze(0).cuda()
    embeds = av_embed(ids_t).float()
    v = normalize_activation(activation.float().view(1, -1), av_cfg.injection_scale)
    embeds = inject_at_marked_positions(
        ids_t.cpu(), embeds.cpu(), v,
        av_cfg.injection_token_id,
        av_cfg.injection_left_neighbor_id,
        av_cfg.injection_right_neighbor_id,
    ).to(av.device, torch.bfloat16)
    out = av.generate(inputs_embeds=embeds, max_new_tokens=max_new_tokens, do_sample=False,
                      pad_token_id=av_tok.pad_token_id or av_tok.eos_token_id)
    raw = av_tok.decode(out[0], skip_special_tokens=True)
    m = EXPL_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()

def cos(a, b):
    return (a @ b / (a.norm() * b.norm())).item()

# ─── 0. Quick cos diagnostic on the new (prepended, content-style) steer ──

print("\n=== diagnostic: does the new steer move AR's encoding? ===")
diag_prompts = ["The most famous landmark in Paris is",
                "Tell me a fun fact about photosynthesis: "]
for p in diag_prompts:
    a = get_resid(p)
    v = av_verbalize(a)
    a_recon  = ar.reconstruct(v)
    a_steer  = ar.reconstruct(STEER + v)            # prepended now
    print(f"  {p!r}")
    print(f"    cos(a_recon, a_steer)  = {cos(a_recon, a_steer):+.4f}")
    print(f"    ‖Δ‖/‖a_recon‖          = {(a_steer - a_recon).norm() / a_recon.norm():.3f}")

print("\nProceeding to full per-token steered generation…")

# ─── 1. Per-token steered generation ──────────────────────────────────────

@torch.inference_mode()
def per_token_complete(prompt, max_new_tokens=20):
    enc = base_tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc.input_ids.cuda()
    am = enc.attention_mask.cuda()
    state = {"step": 0}

    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if hs.shape[1] != 1:
            return output
        last_resid = hs[0, -1, :].float().cpu()
        orig_norm = last_resid.norm().clamp_min(1e-12)
        v = av_verbalize(last_resid)
        a_new = ar.reconstruct(STEER + v)            # PREPEND content steer
        a_new = a_new / a_new.norm().clamp_min(1e-12) * orig_norm
        new_hs = hs.clone()
        new_hs[0, -1, :] = a_new.to(hs.dtype).to(hs.device)
        state["step"] += 1
        if state["step"] in (1, 5, 10, 15) or state["step"] % 5 == 0:
            print(f"    [tok {state['step']:2d}] AV: {' '.join(v.split())[:120]}…")
        return (new_hs, *output[1:]) if is_tuple else new_hs

    h = base_text.layers[LAYER_K].register_forward_hook(hook)
    t0 = time.time()
    try:
        out = base.generate(input_ids=ids, attention_mask=am,
                            max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=base_tok.pad_token_id or base_tok.eos_token_id)
    finally:
        h.remove()
    print(f"    [{time.time()-t0:.0f}s for {state['step']} patched tokens]")
    return base_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

prompts = ["The most famous landmark in Paris is",
           "Tell me a fun fact about photosynthesis: "]

for p in prompts:
    print("\n" + "=" * 78)
    print(f"PROMPT: {p!r}")
    print("=" * 78)
    print("--- STEERED v2 (long content prepend, every decode token) ---")
    out = per_token_complete(p, max_new_tokens=20)
    print(f"\n  -> {out!r}")
