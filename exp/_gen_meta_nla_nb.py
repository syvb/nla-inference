"""Generate exp/meta_nla.ipynb. Run: python3 exp/_gen_meta_nla_nb.py"""
import json, pathlib

cells = []

def md(*lines):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _src(lines)})

def code(*lines):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": _src(lines)})

def _src(lines):
    # join with newlines; ipynb wants a list of strings each ending in \n
    text = "\n".join(lines)
    out = text.split("\n")
    return [l + "\n" for l in out[:-1]] + [out[-1]]


md(
"# Meta-NLAs: running an NLA's verbalizer (AV) on the AV itself",
"",
"An **NLA** (Natural Language Autoencoder, [Fraser-Taliente et al. 2026](https://transformer-circuits.pub/2026/nla/))",
"is two fine-tuned copies of a target model `M`:",
"",
"- **AV** (activation verbalizer): `activation vector → text`. The vector is injected as a single",
"  token-embedding into a fixed prompt; the AV autoregresses an `<explanation>`.",
"- **AR** (activation reconstructor): `text → activation vector` (truncated `K+1`-layer LM + linear head).",
"",
"The key fact this notebook exploits: **the AV is itself a copy of `M`**, so while the AV is busy",
"verbalizing an injected vector, its *own* layer-`L` residual stream carries activations that live in the",
"*same space* the NLA was trained to read. That makes a self-application well-defined:",
"",
"> **Meta-NLA.** Inject an activation `h` into the AV and let it write an explanation. While it does so,",
"> snapshot the AV's own layer-`L` activation `h'` (at the injection site, the last prompt token, or any",
"> generated token). Now feed `h'` *back into the AV*. The resulting text is a verbalization of what the AV",
"> was internally representing *while it verbalized* — a description of the verbalizer's own cognition.",
"",
"You can iterate this (`h → h' → h'' → …`) and ask whether it converges, whether the AV's internal state",
"while explaining is itself interpretable, and how it relates to the original `h`.",
"",
"This notebook runs the AV as a plain HuggingFace model (not the SGLang server in the repo) precisely",
"because we need `output_hidden_states` to read the AV's own activations — something the inference server",
"does not expose.",
)

md(
"## 0. Runtime",
"",
"Use a **GPU runtime with ≥ 24 GB** (Colab A100 ideal; L4 works if you skip or CPU-offload the AR).",
"`Runtime → Change runtime type → A100 / L4`. The default model below is **Qwen2.5-7B** (ungated, ~15 GB",
"in bf16). Gemma/Llama NLAs need `HF_TOKEN` and more memory.",
)

code(
"!nvidia-smi -L || echo 'NO GPU — set Runtime -> Change runtime type -> GPU'",
)

code(
"# torch ships with Colab; install the rest. (httpx/orjson are imported by nla_inference.py.)",
"!pip install -q -U transformers safetensors pyyaml httpx orjson accelerate huggingface_hub",
)

code(
"# Pull the repo so we can reuse its battle-tested injection / config helpers.",
"import os, sys",
"if not os.path.isdir('nla-inference'):",
"    !git clone --depth 1 https://github.com/syvb/nla-inference.git",
"sys.path.insert(0, 'nla-inference')",
"import nla_inference as nlai   # load_nla_config, normalize_activation, inject_at_marked_positions, NLACritic",
"print('helpers:', [n for n in ('load_nla_config','normalize_activation','inject_at_marked_positions','NLACritic') if hasattr(nlai, n)])",
)

md(
"## 1. Config",
"",
"`LAYER` is the NLA extraction layer (Qwen2.5-7B → 20). HF `output_hidden_states` returns",
"`hidden_states[0]=embeddings, hidden_states[i]=output of block i`, so the layer-`L` residual stream the",
"NLA reads is `hidden_states[LAYER]`.",
)

code(
"import torch, numpy as np",
"",
"AV_REPO = 'kitft/nla-qwen2.5-7b-L20-av'   # verbalizer",
"AR_REPO = 'kitft/nla-qwen2.5-7b-L20-ar'   # reconstructor (optional, for fidelity scoring)",
"LAYER   = 20                              # extraction layer L",
"DEVICE  = 'cuda'",
"DTYPE   = torch.bfloat16",
"",
"# For gated Gemma/Llama NLAs, log in first:",
"# from huggingface_hub import login; login('hf_...')",
"",
"from huggingface_hub import snapshot_download",
"av_dir = snapshot_download(AV_REPO)",
"print('AV checkpoint at', av_dir)",
)

md(
"## 2. The AV as a HuggingFace model",
"",
"`MetaAV` wraps the verbalizer with three operations:",
"",
"- `verbalize(h)` — inject `h`, generate the explanation, **and** return the AV's own layer-`L` activation",
"  at every position (prompt + generated tokens). This is what makes the meta-loop possible.",
"- `extract_text(text)` — a plain forward pass (no injection) to get layer-`L` activations of ordinary text,",
"  a convenient source of `h` to seed experiments.",
"- injection reuses `nla_inference.normalize_activation` + `inject_at_marked_positions`, and uses the model's",
"  own embedding module so the architecture embed-scale (√d for Gemma) is handled for free.",
)

code(
"from transformers import AutoTokenizer, AutoModelForCausalLM",
"",
"class MetaAV:",
"    def __init__(self, ckpt_dir, layer, device=DEVICE, dtype=DTYPE):",
"        self.tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)",
"        self.cfg = nlai.load_nla_config(ckpt_dir, self.tok)   # asserts tokenizer/template match",
"        self.model = AutoModelForCausalLM.from_pretrained(",
"            ckpt_dir, torch_dtype=dtype, trust_remote_code=True).to(device).eval()",
"        self.embed = self.model.get_input_embeddings()        # applies arch embed-scale internally",
"        self.layer, self.device = layer, device",
"        if self.tok.pad_token_id is None:",
"            self.tok.pad_token = self.tok.eos_token",
"",
"    def _prompt(self, v=None):",
"        content = self.cfg.actor_prompt_template.format(injection_char=self.cfg.injection_char)",
"        ids = self.tok.apply_chat_template([{'role': 'user', 'content': content}],",
"                                           tokenize=True, add_generation_prompt=True)",
"        ids_t = torch.tensor(ids, device=self.device)[None]   # [1, T]",
"        with torch.no_grad():",
"            e = self.embed(ids_t)                             # [1, T, d]",
"        inj_pos = (ids_t[0] == self.cfg.injection_token_id).nonzero().flatten().tolist()",
"        inj_pos = inj_pos[0] if inj_pos else None",
"        if v is not None:",
"            vt = torch.as_tensor(np.asarray(v, np.float32)).view(1, -1)",
"            vt = nlai.normalize_activation(vt, self.cfg.injection_scale)   # → injection_scale L2-norm",
"            e = nlai.inject_at_marked_positions(",
"                ids_t, e, vt, self.cfg.injection_token_id,",
"                self.cfg.injection_left_neighbor_id, self.cfg.injection_right_neighbor_id)",
"        return ids_t, e, inj_pos",
"",
"    @torch.no_grad()",
"    def verbalize(self, v, max_new_tokens=220, temperature=1.0, **samp):",
"        ids_t, e, inj_pos = self._prompt(v)",
"        attn = torch.ones(e.shape[:2], dtype=torch.long, device=self.device)",
"        gen = self.model.generate(",
"            inputs_embeds=e, attention_mask=attn, max_new_tokens=max_new_tokens,",
"            do_sample=temperature > 0, temperature=max(temperature, 1e-5),",
"            pad_token_id=self.tok.pad_token_id, **samp)",
"        new_ids = gen[0]                                      # inputs_embeds → only NEW tokens returned",
"        raw = self.tok.decode(new_ids, skip_special_tokens=False)",
"        m = nlai.EXPLANATION_RE.search(raw)",
"        expl = m.group(1).strip() if m else raw",
"        # Re-run the full (prompt+generated) sequence once to read layer-L activations everywhere.",
"        full_e = torch.cat([e, self.embed(new_ids[None])], dim=1)",
"        full_attn = torch.ones(full_e.shape[:2], dtype=torch.long, device=self.device)",
"        hs = self.model(inputs_embeds=full_e, attention_mask=full_attn,",
"                        output_hidden_states=True).hidden_states[self.layer][0]   # [T_full, d]",
"        toks = (self.tok.convert_ids_to_tokens(ids_t[0].tolist())",
"                + self.tok.convert_ids_to_tokens(new_ids.tolist()))",
"        if inj_pos is not None:",
"            toks[inj_pos] = '\\u2039INJECT\\u203a'",
"        return dict(explanation=expl, raw=raw, hidden=hs.float().cpu(), tokens=toks,",
"                    prompt_len=ids_t.shape[1], inject_pos=inj_pos, new_ids=new_ids.cpu())",
"",
"    @torch.no_grad()",
"    def extract_text(self, text):",
"        ids = self.tok(text, return_tensors='pt').input_ids.to(self.device)",
"        hs = self.model(ids, output_hidden_states=True).hidden_states[self.layer][0]",
"        return hs.float().cpu(), self.tok.convert_ids_to_tokens(ids[0].tolist())",
"",
"av = MetaAV(av_dir, LAYER)",
"print('loaded AV  d_model=%d  inj_scale=%s' % (av.cfg.d_model, av.cfg.injection_scale))",
)

md(
"### Sanity check",
"",
"Inject a random vector and an activation extracted from real text. Output should be **fluent English**",
"`<explanation>` text. All-CJK output or text describing a CJK character ⇒ injection failed (wrong scale /",
"template drift) — see the repo README's *Debugging* section.",
)

code(
"r = av.verbalize(np.random.randn(av.cfg.d_model), max_new_tokens=120)",
"print('RANDOM VECTOR →\\n', r['explanation'][:600])",
"",
"hs, toks = av.extract_text('The mitochondrion is the powerhouse of the cell.')",
"r2 = av.verbalize(hs[-1].numpy(), max_new_tokens=120)",
"print('\\nREAL ACTIVATION (last token) →\\n', r2['explanation'][:600])",
)

md(
"## 3. (Optional) the AR, for fidelity scoring",
"",
"The AR reconstructs an activation from explanation text; `cos(reconstruction, original)` measures how well",
"the AV's words captured the vector (`≈0.9` good, `0.5` mediocre, `0` orthogonal). Skip this cell on a",
"tight-memory GPU — the meta-loop works without it.",
)

code(
"USE_AR = True",
"critic = None",
"if USE_AR:",
"    ar_dir = snapshot_download(AR_REPO)",
"    critic = nlai.NLACritic(ar_dir, device=DEVICE)",
"",
"def cos_of(expl, h):",
"    return None if critic is None else round(critic.score(expl, np.asarray(h, np.float32))[1], 3)",
)

md(
"## 4. The meta-loop",
"",
"`meta_chain` is the core experiment. Starting from `h0`, at each level it verbalizes the current",
"activation, then takes the AV's *own* layer-`L` activation (at position `where`) as the next activation.",
"",
"`where` selects which of the AV's internal activations to recurse on:",
"",
"- `'inject'` — the injection site **after** `LAYER` blocks: how the AV has re-encoded the vector you gave it.",
"- `'last_prompt'` — the final prompt token, just before the AV starts writing.",
"- `'last'` — the last generated token (the AV's state at the end of its explanation).",
"- an `int` — an absolute position into `hidden` (e.g. a specific generated word).",
)

code(
"def pos_of(res, where):",
"    if isinstance(where, int):     return where",
"    if where == 'inject':          return res['inject_pos']",
"    if where == 'last_prompt':     return res['prompt_len'] - 1",
"    if where == 'last':            return res['hidden'].shape[0] - 1",
"    raise ValueError(where)",
"",
"def meta_chain(h0, levels=4, where='last', max_new_tokens=200, temperature=1.0, **samp):",
"    rows, h = [], np.asarray(h0, np.float32)",
"    for lvl in range(levels):",
"        res = av.verbalize(h, max_new_tokens=max_new_tokens, temperature=temperature, **samp)",
"        p = pos_of(res, where)",
"        rows.append(dict(level=lvl, where=where, pos=p, cos=cos_of(res['explanation'], h),",
"                         token=res['tokens'][p], explanation=res['explanation'], res=res))",
"        h = res['hidden'][p].numpy()        # the AV's OWN activation → next input",
"    return rows",
"",
"def show(rows, width=560):",
"    for r in rows:",
"        print('\\u2500' * 92)",
"        print(f\"level {r['level']}  where={r['where']!r}  pos={r['pos']} (tok {r['token']!r})  cos={r['cos']}\")",
"        print(r['explanation'][:width])",
"    print('\\u2500' * 92)",
)

md(
"### Run it",
"",
"Seed with a real activation, then recurse on the AV's end-of-explanation state. Read the chain top to",
"bottom: level 0 is the ordinary NLA explanation of your text; deeper levels describe the verbalizer's own",
"internal state as it explains.",
)

code(
"hs, toks = av.extract_text(",
"    'Breaking: the central bank raised interest rates today, citing persistent inflation.')",
"h0 = hs[-1].numpy()",
"chain = meta_chain(h0, levels=4, where='last')",
"show(chain)",
)

md(
"## 5. Things to try",
"",
"The cells below are starting points — edit freely.",
)

md(
"**(a) Verbalize how the AV re-encoded *your* vector** (`where='inject'`). Compare level 0 (the AV's words)",
"to level 1 (a description of the AV's internal representation of the same vector at layer `L`). Do they",
"agree?",
)

code(
"show(meta_chain(h0, levels=2, where='inject'), width=600)",
)

md(
"**(b) Does the loop converge?** Track reconstruction cosine and explanation length across many levels for",
"different `where` sites. A rising/stabilising `cos` suggests the AV's internal state settles into a",
"self-consistent, well-verbalizable attractor; a collapse suggests drift toward the training prior.",
)

code(
"import textwrap",
"for where in ['last', 'last_prompt', 'inject']:",
"    chain = meta_chain(h0, levels=6, where=where, max_new_tokens=160)",
"    coss = [r['cos'] for r in chain]",
"    lens = [len(r['explanation']) for r in chain]",
"    print(f'where={where:<12} cos per level = {coss}   |expl| = {lens}')",
)

md(
"**(c) Close the autoencoder loop instead.** Rather than the AV's *internal* activation, recurse on the",
"AR's *reconstruction* of the AV's explanation: `h → AV → text → AR → ĥ → AV → …`. This is the pure",
"text↔vector fixed point of the NLA, a useful contrast to the meta-loop above.",
)

code(
"def ar_chain(h0, levels=4, max_new_tokens=200):",
"    assert critic is not None, 'needs the AR (section 3)'",
"    rows, h = [], np.asarray(h0, np.float32)",
"    for lvl in range(levels):",
"        res = av.verbalize(h, max_new_tokens=max_new_tokens)",
"        h_hat = critic.reconstruct(res['explanation']).numpy()",
"        rows.append(dict(level=lvl, where='AR', pos=-1, token='AR',",
"                         cos=cos_of(res['explanation'], h), explanation=res['explanation'], res=res))",
"        h = h_hat",
"    return rows",
"",
"if critic is not None:",
"    show(ar_chain(h0, levels=4))",
)

md(
"**(d) Bring your own activation.** Swap in any text, a specific token position, a steering vector, or a",
"hand-built vector. For activations faithful to the *target* model (the AV is a fine-tuned copy of it, so",
"its activations are close but not identical), load `Qwen/Qwen2.5-7B-Instruct` separately and extract",
"`hidden_states[LAYER]` from it — memory permitting.",
)

code(
"# your_text = 'Once upon a time, in a kingdom by the sea,'",
"# hs, toks = av.extract_text(your_text)",
"# print(list(enumerate(toks)))            # pick a token position",
"# show(meta_chain(hs[12].numpy(), levels=3, where='last'))",
)

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = pathlib.Path(__file__).parent / "meta_nla.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out, "with", len(cells), "cells")
