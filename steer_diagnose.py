"""Quick diagnostic: did the steer string actually survive AR encoding?

Computes cos(a_recon, a_steered) and ||a_steered - a_recon||/||a_recon|| for each
of our three prompts. If cos ≈ 1, AR ignored the steer text — explanation is
'AR was trained on AV-style summaries, not behavioral instructions, so steer
text reads as noise.' If cos < 0.95, the encoding shifted but the patch wasn't
strong enough to redirect the model — explanation is 'one-position patch is too
weak.'
"""
import torch
from nla_inference import NLACritic
from transformers import (
    AutoTokenizer, BitsAndBytesConfig,
    Gemma3ForConditionalGeneration, Gemma3ForCausalLM,
)
from nla_inference import normalize_activation, inject_at_marked_positions, load_nla_config

BASE_DIR = "/workspace/nla-models/base"
AV_DIR   = "/workspace/nla-models/av"
AR_DIR   = "/workspace/nla-models/ar"
LAYER_K  = 32

STEER = (
    " Behavior note: the model is obsessed with the Golden Gate Bridge, and stops "
    "at nothing to always mention the Golden Gate Bridge as much as possible in all circumstances."
)

print("Loading base 8-bit + AV 4-bit + AR cpu…")
bnb_8 = BitsAndBytesConfig(load_in_8bit=True)
bnb_4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)

base = Gemma3ForConditionalGeneration.from_pretrained(
    BASE_DIR, quantization_config=bnb_8, device_map="cuda:0", dtype=torch.bfloat16,
).eval()
base_tok = AutoTokenizer.from_pretrained(BASE_DIR)
base_text = base.language_model.model if not hasattr(base.language_model, "layers") else base.language_model

av = Gemma3ForCausalLM.from_pretrained(
    AV_DIR, quantization_config=bnb_4, device_map="cuda:0", dtype=torch.bfloat16,
).eval()
av_tok = AutoTokenizer.from_pretrained(AV_DIR)
av_cfg = load_nla_config(AV_DIR, av_tok)
av_embed = av.get_input_embeddings()

ar = NLACritic(AR_DIR, device="cpu")

import re
EXPL_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

@torch.inference_mode()
def get_resid(text):
    enc = base_tok(text, return_tensors="pt", add_special_tokens=True)
    out = base_text(enc.input_ids.cuda(), attention_mask=enc.attention_mask.cuda(),
                    output_hidden_states=True, use_cache=False)
    return out.hidden_states[LAYER_K + 1][0, -1].float().cpu()

@torch.inference_mode()
def av_verbalize(activation):
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
    out = av.generate(inputs_embeds=embeds, max_new_tokens=180, do_sample=False,
                      pad_token_id=av_tok.pad_token_id or av_tok.eos_token_id)
    raw = av_tok.decode(out[0], skip_special_tokens=True)
    m = EXPL_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()

prompts = [
    "The most famous landmark in Paris is",
    "Tell me a fun fact about photosynthesis: ",
    "List three popular vacation destinations in the United States: ",
]

def cos(a, b):
    return (a @ b / (a.norm() * b.norm())).item()

for p in prompts:
    a = get_resid(p)
    v = av_verbalize(a)
    v_steer = v + STEER

    a_recon = ar.reconstruct(v)
    a_steer = ar.reconstruct(v_steer)

    # Mentions of "Golden Gate" in either verbalization?
    print(f"\nPROMPT: {p!r}")
    print(f"  ||a||={a.norm().item():.0f}  ||a_recon||={a_recon.norm().item():.2f}  ||a_steer||={a_steer.norm().item():.2f}")
    print(f"  cos(a, a_recon)         = {cos(a, a_recon):+.4f}")
    print(f"  cos(a_recon, a_steer)   = {cos(a_recon, a_steer):+.4f}   (1.00 = AR ignored the steer text)")
    print(f"  cos(a, a_steer)         = {cos(a, a_steer):+.4f}")
    delta = (a_steer - a_recon).norm().item() / a_recon.norm().item()
    print(f"  ||delta|| / ||a_recon|| = {delta:.3f}")
