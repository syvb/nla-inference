"""Real activations: feed text through base Gemma-3-12B-IT, extract layer-32
residual at last token, inject into AV (via sglang) → verbalize. Score with AR.

This is the faithful round-trip:
  text  -[base model]->  layer-32 residual  -[AV]->  English verbalization
                                              -[AR]->  predicted activation
                                              cos(predicted, original) ≈ 0.9 = good
"""
import gc
import os
import numpy as np
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from nla_inference import NLAClient, NLACritic

BASE_DIR = "/workspace/nla-models/base"
AV_DIR   = "/workspace/nla-models/av"
AR_DIR   = "/workspace/nla-models/ar"
LAYER_K  = 32   # extraction layer (per av/nla_meta.yaml: extraction_layer_index)

texts = [
    "The Eiffel Tower in Paris is an iron lattice tower built in 1889.",
    "Python is a high-level programming language widely used for data science.",
    "The mitochondria generate ATP through cellular respiration in eukaryotic cells.",
    "Beethoven composed nine symphonies during the Classical-to-Romantic transition.",
    "The recipe calls for two cups of flour, one teaspoon of salt, and a half cup of sugar.",
    "King Henry VIII of England had six wives and broke from the Catholic Church.",
    "Quantum entanglement allows two particles to share state across arbitrary distances.",
]

print(f"Loading base Gemma-3-12B-IT on CPU (will truncate to first {LAYER_K+1} layers)…")
base = Gemma3ForConditionalGeneration.from_pretrained(
    BASE_DIR,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu",
)
base.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)

# Find the text stack: .language_model can be either Gemma3ForCausalLM (older
# transformers) or Gemma3TextModel directly (newer). Handle both.
lm = base.language_model
text_model = lm.model if hasattr(lm, "layers") is False and hasattr(lm, "model") else lm

# Truncate layers and bypass final LN — we want the raw residual stream after
# block K, not the LN'd version (matches AR's training-time extraction).
n_keep = LAYER_K + 1
text_model.layers = torch.nn.ModuleList(list(text_model.layers[:n_keep]))
text_model.norm = torch.nn.Identity()
gc.collect()

print(f"Truncated to {len(text_model.layers)} layers. Free RAM after GC:", end=" ")
os.system("free -g | awk '/Mem:/ {print $7\" GB available\"}'")
print()

# AV via sglang (already running) and AR via CPU.
av = NLAClient(AV_DIR, sglang_url="http://localhost:30000")
ar = NLACritic(AR_DIR, device="cpu")

print()
print("=" * 78)
print("text -> base Gemma layer-32 residual @ last token -> AV.generate -> verbalize")
print("=" * 78)
for i, text in enumerate(texts):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    with torch.no_grad():
        out = text_model(ids, use_cache=False)
    activation = out.last_hidden_state[0, -1].float().cpu().numpy()  # post-block-K residual
    norm = float(np.linalg.norm(activation))
    last_tok_str = tokenizer.decode([ids[0, -1].item()])

    av_out = av.generate(activation, temperature=0.0, max_new_tokens=200)
    rt_mse, rt_cos = ar.score(av_out, activation)

    print(f"\n[{i}] INPUT: {text}")
    print(f"    last token: {last_tok_str!r}   ||a||={norm:.0f}")
    print(f"    round-trip cos={rt_cos:+.3f}  mse={rt_mse:.3f}")
    print(f"    AV-> {av_out}")
