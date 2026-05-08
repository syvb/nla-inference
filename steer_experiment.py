"""Steering experiment.

For each prompt, generate three completions from base Gemma-3-12B-IT:
  A: vanilla (no patch)                           — reference
  B: layer-32 residual @ last prompt token replaced with AR(AV(a))     — round-trip drift
  C: same as B but with a steering string appended to AV's verbalization
     before AR re-encodes it                                            — steered

Compare A→B (how lossy the autoencoder round-trip is) vs B→C (whether language
patched into the AV verbalization survives AR encoding and changes behavior).

All on GPU at 4-bit (base + AV; AR stays on CPU bf16).
"""
import re
import gc
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

print("Loading base Gemma-3-12B-IT (8-bit, bf16)…")
base = Gemma3ForConditionalGeneration.from_pretrained(
    BASE_DIR, quantization_config=bnb_8, device_map="cuda:0",
    dtype=torch.bfloat16,
)
base.eval()
base_tok = AutoTokenizer.from_pretrained(BASE_DIR)
lm = base.language_model
base_text = lm.model if not hasattr(lm, "layers") else lm
print(f"  base.text layers: {len(base_text.layers)}")

print("Loading AV (4-bit, bf16)…")
av = Gemma3ForCausalLM.from_pretrained(
    AV_DIR, quantization_config=bnb_4, device_map="cuda:0",
    dtype=torch.bfloat16,
)
av.eval()
av_tok = AutoTokenizer.from_pretrained(AV_DIR)
av_cfg = load_nla_config(AV_DIR, av_tok)
av_embed = av.get_input_embeddings()  # Gemma3TextScaledWordEmbedding (forward applies √d)
print("  AV ready.")

print("Loading AR (CPU bf16)…")
ar = NLACritic(AR_DIR, device="cpu")

# ─── Activation extraction / verbalization helpers ─────────────────────────

EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

@torch.inference_mode()
def get_layer_K_residual(text: str):
    enc = base_tok(text, return_tensors="pt", add_special_tokens=True)
    ids = enc.input_ids.cuda()
    am = enc.attention_mask.cuda()
    out = base_text(ids, attention_mask=am, output_hidden_states=True, use_cache=False)
    resid = out.hidden_states[LAYER_K + 1][0, -1].float().cpu()
    return ids, resid


@torch.inference_mode()
def av_verbalize(activation: torch.Tensor, max_new_tokens: int = 180) -> str:
    """Mimic NLAClient.generate via HF transformers."""
    content = av_cfg.actor_prompt_template.format(injection_char=av_cfg.injection_char)
    input_ids = av_tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()
    embeds = av_embed(ids_t).float()  # ScaledWordEmbedding handles √d internally

    v_scaled = normalize_activation(
        activation.float().view(1, -1), av_cfg.injection_scale
    )
    embeds = inject_at_marked_positions(
        ids_t.cpu(), embeds.cpu(), v_scaled,
        av_cfg.injection_token_id,
        av_cfg.injection_left_neighbor_id,
        av_cfg.injection_right_neighbor_id,
    ).to(av.device, torch.bfloat16)

    out = av.generate(
        inputs_embeds=embeds, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=av_tok.pad_token_id or av_tok.eos_token_id,
    )
    raw = av_tok.decode(out[0], skip_special_tokens=True)
    m = EXPL_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()


class ResidualPatch:
    """Replace layer-K residual at one position during prefill via forward hook."""
    def __init__(self, layer_module):
        self.layer = layer_module
        self.handle = None
        self.position = None
        self.vector = None

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def set(self, position: int, vector: torch.Tensor):
        self.position = position
        self.vector = vector

    def _hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        # Only patch during prefill (the call where T > 1). Decode steps have T=1
        # — by then position-of-interest is in KV cache and untouched.
        if hs.shape[1] <= 1 or self.vector is None:
            return output
        hs = hs.clone()
        hs[:, self.position, :] = self.vector.to(hs.dtype).to(hs.device)
        return (hs, *output[1:]) if is_tuple else hs


@torch.inference_mode()
def complete(prompt: str, patch_vec: torch.Tensor | None, max_new_tokens: int = 80) -> str:
    enc = base_tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc.input_ids.cuda()
    am = enc.attention_mask.cuda()
    last_pos = ids.shape[1] - 1

    with ResidualPatch(base_text.layers[LAYER_K]) as patcher:
        if patch_vec is not None:
            patcher.set(last_pos, patch_vec)
        out = base.generate(
            input_ids=ids, attention_mask=am,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=base_tok.pad_token_id or base_tok.eos_token_id,
        )
    return base_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ─── Run ───────────────────────────────────────────────────────────────────

prompts = [
    "The most famous landmark in Paris is",
    "Tell me a fun fact about photosynthesis: ",
    "List three popular vacation destinations in the United States: ",
]

for prompt in prompts:
    print("\n" + "=" * 78)
    print(f"PROMPT: {prompt!r}")
    print("=" * 78)

    # A: vanilla
    out_a = complete(prompt, patch_vec=None)
    print(f"\n[A] vanilla completion:\n   {out_a!r}")

    # Extract real layer-32 residual at last token
    _ids, resid = get_layer_K_residual(prompt)
    orig_norm = resid.norm().item()
    print(f"\n  layer-{LAYER_K} residual ||a||={orig_norm:.0f}")

    # Verbalize
    v = av_verbalize(resid)
    v_steered = v + STEER
    print(f"\n  AV verbalization: {v}")
    print(f"\n  steered (added):  …{STEER.strip()[:120]}…")

    # AR reconstructs both back to activation vectors
    a_recon = ar.reconstruct(v)
    a_steer = ar.reconstruct(v_steered)

    # AR returns at norm ~√d (mse_scale-normalized training); rescale to the
    # original residual's norm so the rest of layers 33-47 see comparable
    # magnitudes (otherwise downstream attention silently blows up or vanishes).
    def _rescale(x):
        return x / x.norm().clamp_min(1e-12) * orig_norm

    a_recon_s = _rescale(a_recon)
    a_steer_s = _rescale(a_steer)

    # B: round-trip patched
    out_b = complete(prompt, patch_vec=a_recon_s)
    print(f"\n[B] AR(AV(a)) patched (round-trip drift):\n   {out_b!r}")

    # C: steered
    out_c = complete(prompt, patch_vec=a_steer_s)
    print(f"\n[C] AR(AV(a) + STEER) patched:\n   {out_c!r}")
