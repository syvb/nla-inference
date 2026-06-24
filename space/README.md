---
title: NLA Image Soft-Token Verbalizer (Gemma-3)
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
suggested_hardware: zero-a10g
short_description: Verbalize Gemma-3 image soft tokens with the NLA AV
---

# NLA image soft-token verbalizer

Upload an image. The multimodal base model (`google/gemma-3-*-it`) encodes it
into 256 image **soft tokens** and runs them through its residual stream; the
matching **NLA activation verbalizer** (AV) then reads back, in natural
language, what each soft-token vector "means".

This is an **out-of-distribution probe**: the AV was trained to verbalize
*text*-derived residual activations at a single layer, never vision soft tokens.
Garbage / CJK output is an expected failure mode — the point is to look.

## How it works

1. **Extract** — `Gemma3ForConditionalGeneration` encodes the image; we read
   `hidden_states[depth]` at the 256 image-token positions. `depth=0` is the raw
   soft tokens (projector output); `depth = av_layer+1` matches the AV's training
   layer.
2. **Verbalize** — each `[d_model]` vector is L2-renormalized to the AV's
   `injection_scale` and injected into the AV's fixed prompt at the marker token,
   then autoregressed into an `<explanation>` (the `nla_inference.py` recipe,
   run locally + batched instead of via SGLang).

The two models never sit on the GPU together (base off → AV on), so the 27B pair
fits in 70 GB.

## Configuration

- **`NLA_MODEL`** (Space variable): `12b` (default) or `27b` — selects which
  base+AV pair to load.
- **`HF_TOKEN`** (Space secret): required — `google/gemma-3-*-it` is gated.

Hardware: **ZeroGPU**.
