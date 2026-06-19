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
"# Versions PINNED to a combination verified end-to-end for these checkpoints: the injection",
"# char ㈎ must tokenize to a single id (149705) in context, which is tokenizers-version-sensitive.",
"!pip install -q -U 'transformers==5.8.1' 'tokenizers==0.22.2' safetensors pyyaml httpx orjson accelerate huggingface_hub",
)

code(
"# Pull the repo for its injection helpers (normalize_activation, inject_at_marked_positions,",
"# NLACritic, EXPLANATION_RE). NOTE: we deliberately do NOT use nlai.load_nla_config / one-step",
"# apply_chat_template — see the tokenization note in section 2.",
"import os, sys",
"if not os.path.isdir('nla-inference'):",
"    !git clone --depth 1 https://github.com/syvb/nla-inference.git",
"sys.path.insert(0, 'nla-inference')",
"import nla_inference as nlai",
"print('helpers:', [n for n in ('normalize_activation','inject_at_marked_positions','NLACritic','EXPLANATION_RE') if hasattr(nlai, n)])",
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
"AV_REPO   = 'kitft/nla-qwen2.5-7b-L20-av'   # verbalizer",
"AR_REPO   = 'kitft/nla-qwen2.5-7b-L20-ar'   # reconstructor (optional, for fidelity scoring)",
"BASE_REPO = 'Qwen/Qwen2.5-7B-Instruct'      # source of the chat template (see section 2)",
"LAYER     = 20                              # extraction layer L",
"DEVICE    = 'cuda'",
"DTYPE     = torch.bfloat16",
"",
"# For gated Gemma/Llama NLAs, log in first (also needed for their gated BASE_REPO):",
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
"",
"**Two tokenization gotchas (both handled below):**",
"",
"1. The released checkpoints ship stock vocab files with **no `chat_template`** (the release scrubber copies",
"   `tokenizer*.json` unmodified). transformers 5.x has no default template, so we borrow the canonical one",
"   from `BASE_REPO` — the exact chat format the AV was trained on.",
"2. One-step `apply_chat_template(tokenize=True)` is **broken in transformers 5.x** for these tokenizers",
"   (it returns a truncated 2-token sequence — which is why `nla_inference.load_nla_config` fails with",
"   *\"injection token appears 0×\"*). We use the README-blessed two-step path instead: render to text with",
"   `tokenize=False`, then `encode(..., add_special_tokens=False)`. That keeps the injection char `㈎` as a",
"   single id and avoids double-BOS on Gemma/Llama (Qwen has none).",
"",
"This class is **verified against Qwen2.5-7B** (plain causal LM, embed_scale=1). The Gemma-3 NLAs use a",
"multimodal wrapper (`config.text_config`, `model.language_model`); `get_input_embeddings()` and",
"`output_hidden_states` may need to reach through `.language_model`, and the √d embed-scale path is",
"**unvalidated** there — sanity-check the output is fluent English before trusting it.",
)

code(
"import yaml",
"from transformers import AutoTokenizer, AutoModelForCausalLM",
"",
"def build_av_config(ckpt_dir, tok):",
"    # NLAConfig straight from nla_meta.yaml (we skip nlai.load_nla_config because it uses the",
"    # broken one-step apply_chat_template). Sanity-check the injection char tokenizes to one id.",
"    meta = yaml.safe_load(open(os.path.join(ckpt_dir, 'nla_meta.yaml')))",
"    tks = meta['tokens']",
"    cfg = nlai.NLAConfig(",
"        d_model=meta['d_model'],",
"        injection_char=tks['injection_char'],",
"        injection_token_id=tks['injection_token_id'],",
"        injection_left_neighbor_id=tks['injection_left_neighbor_id'],",
"        injection_right_neighbor_id=tks['injection_right_neighbor_id'],",
"        actor_prompt_template=meta['prompt_templates'].get('av') or meta['prompt_templates']['actor'],",
"        injection_scale=float(meta['extraction']['injection_scale']))",
"    solo = tok.encode(cfg.injection_char, add_special_tokens=False)",
"    assert solo == [cfg.injection_token_id], (",
"        f'injection char {cfg.injection_char!r} -> {solo}, sidecar says [{cfg.injection_token_id}]. '",
"        'Wrong tokenizers version — pin tokenizers==0.22.2.')",
"    return cfg",
"",
"class MetaAV:",
"    def __init__(self, ckpt_dir, layer, base_repo=None, device=DEVICE, dtype=DTYPE):",
"        self.tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)",
"        if self.tok.chat_template is None:                    # released ckpts ship no template",
"            assert base_repo, 'tokenizer has no chat_template — pass base_repo=BASE_REPO'",
"            self.tok.chat_template = AutoTokenizer.from_pretrained(base_repo).chat_template",
"        self.cfg = build_av_config(ckpt_dir, self.tok)",
"        self.model = AutoModelForCausalLM.from_pretrained(",
"            ckpt_dir, torch_dtype=dtype, trust_remote_code=True).to(device).eval()",
"        self.embed = self.model.get_input_embeddings()        # applies arch embed-scale internally",
"        self.layer, self.device = layer, device",
"        if self.tok.pad_token_id is None:",
"            self.tok.pad_token = self.tok.eos_token",
"        eos = self.model.generation_config.eos_token_id",
"        if eos is None:",
"            eos = self.tok.eos_token_id",
"        self._eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])",
"        ids_t, _, inj_pos = self._prompt()                    # validate the injection site once",
"        assert inj_pos is not None, 'injection token not found in canonical prompt'",
"        assert (ids_t[0, inj_pos - 1].item() == self.cfg.injection_left_neighbor_id and",
"                ids_t[0, inj_pos + 1].item() == self.cfg.injection_right_neighbor_id), (",
"            'injection neighbors mismatch — template/tokenizer drift')",
"",
"    def _prompt(self, v=None):",
"        content = self.cfg.actor_prompt_template.format(injection_char=self.cfg.injection_char)",
"        # TWO-STEP tokenization: render to text, then encode(add_special_tokens=False). One-step",
"        # apply_chat_template(tokenize=True) is broken in transformers 5.x for these tokenizers",
"        # (returns ~2 tokens). add_special_tokens=False also avoids double-BOS on Gemma/Llama.",
"        rendered = self.tok.apply_chat_template(",
"            [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)",
"        ids = self.tok.encode(rendered, add_special_tokens=False)",
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
"        kw = dict(do_sample=temperature > 0)                  # pass temperature only when sampling",
"        if temperature > 0:",
"            kw['temperature'] = temperature                  # temperature=0 → greedy",
"        gen = self.model.generate(",
"            inputs_embeds=e, attention_mask=attn, max_new_tokens=max_new_tokens,",
"            pad_token_id=self.tok.pad_token_id, **kw, **samp)",
"        # With inputs_embeds and no input_ids, generate returns ONLY new tokens.",
"        # Guard the assumption (it has been version-sensitive in transformers).",
"        assert gen.shape[1] <= max_new_tokens, (",
"            'generate returned more tokens than max_new_tokens — your transformers '",
"            'version may be echoing prompt tokens; new_ids offsets would be wrong.')",
"        new_ids = gen[0]",
"        # Drop trailing EOS / end-of-turn tokens so where='last' recurses on the AV's",
"        # last CONTENT activation, not the near-constant end-of-turn direction.",
"        while new_ids.numel() and new_ids[-1].item() in self._eos_ids:",
"            new_ids = new_ids[:-1]",
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
"        # Convenience source of arbitrary [d] vectors to seed experiments. Plain",
"        # tokenization (no chat template) — NOT a faithful target-model extraction;",
"        # the AV is a fine-tuned copy of M so these are close but off-distribution.",
"        ids = self.tok(text, return_tensors='pt').input_ids.to(self.device)",
"        hs = self.model(ids, output_hidden_states=True).hidden_states[self.layer][0]",
"        return hs.float().cpu(), self.tok.convert_ids_to_tokens(ids[0].tolist())",
"",
"av = MetaAV(av_dir, LAYER, base_repo=BASE_REPO)",
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
"the AV's words captured the vector (`≈0.9` good, `0.5` mediocre, `0` orthogonal). The meta-loop works",
"without it. On a 40 GB A100 the AV (~15 GB) + AR (~11 GB) fit together; **on a 24 GB L4 set `USE_AR =",
"False`** or you'll OOM.",
)

code(
"USE_AR = True   # set False on a 24 GB GPU (L4): AV + AR together need ~26 GB",
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
