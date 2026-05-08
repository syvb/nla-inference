"""End-to-end NLA smoke test: random vector -> AV verbalize -> AR reconstruct -> MSE/cos.

Eyeball check: AV output should be coherent English describing some semantic
concept, NOT Chinese characters or descriptions of CJK chars (= injection failed).
For random vectors, MSE will be high (no real signal) — that's expected.
"""
import sys
import numpy as np
import torch
from nla_inference import NLAClient, NLACritic

AV_DIR = "/workspace/nla-models/av"
AR_DIR = "/workspace/nla-models/ar"
SGLANG_URL = "http://localhost:30000"

def main():
    print("=" * 70)
    print("Loading AV client (talks to SGLang)…")
    print("=" * 70)
    av = NLAClient(AV_DIR, sglang_url=SGLANG_URL)
    d = av.cfg.d_model
    print(f"d_model = {d}")

    # Two test vectors:
    # 1) Random unit vector (rescaled to injection_scale by client) - meaningless
    # 2) The embedding of a semantic word from the AV's own vocab, scaled up to
    #    injection_scale - more likely to produce coherent semantic decode
    rng = np.random.default_rng(42)
    v_random = rng.standard_normal(d).astype(np.float32)

    # Use the embedding of "elephant" (token id) as a proxy "real" activation —
    # NB: this is an embedding, not a residual stream activation, but it should
    # give the actor *some* semantic signal to verbalize and is a stronger smoke
    # test than pure noise.
    tok = av.tokenizer
    el_ids = tok.encode("elephant", add_special_tokens=False)
    print(f"'elephant' token ids: {el_ids}")
    v_word = av.embed.weight[el_ids[0]].float().cpu().numpy()
    print(f"raw 'elephant' embedding norm (pre-rescale): {np.linalg.norm(v_word):.3f}")

    vectors = [("random_gaussian", v_random), ("'elephant' embedding", v_word)]

    print()
    print("=" * 70)
    print("AV: vector -> text  (deterministic with temperature=0)")
    print("=" * 70)
    av_outputs = []
    for name, v in vectors:
        print(f"\n--- {name} ---")
        text = av.generate(v, temperature=0.0, max_new_tokens=200)
        print(f"OUTPUT: {text!r}")
        av_outputs.append(text)
        # Quick injection-failure smell check: all-CJK output
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(1, len(text))
        if non_ascii_ratio > 0.5:
            print(f"WARNING: {non_ascii_ratio:.1%} non-ASCII chars — possible injection failure")

    print()
    print("=" * 70)
    print("Loading AR critic onto GPU…")
    print("=" * 70)
    # cuda:0 — but check there's room. After sglang takes ~half the card with
    # FP8 weights, AR (8.4B at bf16 = ~17GB) won't fit. Fall back to CPU if so.
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU free: {free/1e9:.2f}/{total/1e9:.2f} GB")
    if free < 18e9:
        print("Not enough GPU room for AR (needs ~17GB bf16) — loading on CPU.")
        device = "cpu"
    else:
        device = "cuda:0"
    ar = NLACritic(AR_DIR, device=device)

    print()
    print("=" * 70)
    print("Round-trip: AR(AV(v)) vs v  — MSE=2(1-cos), range [0,4]")
    print("=" * 70)
    for (name, v), text in zip(vectors, av_outputs):
        mse, cos = ar.score(text, v)
        print(f"\n--- {name} ---")
        print(f"  AV text:   {text[:120]!r}{'…' if len(text) > 120 else ''}")
        print(f"  MSE: {mse:.4f}    cos: {cos:.4f}")
        # Reference: cos=0.9 → MSE=0.2 (good), cos=0.5 → MSE=1.0 (mediocre)

    print("\nSmoke test complete.")

if __name__ == "__main__":
    main()
