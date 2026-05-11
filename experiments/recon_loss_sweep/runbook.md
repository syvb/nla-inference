# Runbook — reconstruction-loss sweep

Live log of commands and outcomes. Times are UTC.

## 2026-05-10

### 15:14 — Local prep
- Reviewed `recon_loss_sweep.py` (extract / decode / smoke commands).
- Local disk: 277 GB free in `/home/smitop2/`. Plenty.
- Ran `python recon_loss_sweep.py smoke` → all green.

### 15:18 — Pod creation
- POST `https://rest.runpod.io/v1/pods` with:
  - image `runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204`
  - 1× H100 80GB HBM3 (any cloud)
  - 150 GB container disk, 0 GB volume
  - SSH key + `HF_TOKEN` env
- Got pod `abfk12qdlxwyvp`, machine `njcwpdy0mv3v`, US-NE-1 (secure cloud),
  $2.99/hr.
- SSH on `216.243.220.227:16974`, NVIDIA driver 580.126.09.

### 15:21 — Environment
- Pre-installed: torch 2.9.1+cu129, httpx.
- Installed: sglang 0.5.11, transformers 5.6.0, datasets 4.8.5,
  pyarrow 24.0.0, numpy 2.3.5, orjson, safetensors, pyyaml, accelerate,
  hf_transfer.
- scp'd `recon_loss_sweep.py` and `nla_inference.py` into `/workspace/`.

### 15:30–15:45 — Download false start, then real start
- First download attempt used `huggingface-cli download` — that command is now
  deprecated and exits with a "use `hf` instead" hint, so logs looked fine but
  no bytes moved. Patched `pod_runner.sh` to use `hf download`.
- Second attempt: AV / AR partials started flowing (~10 GB in), but
  `gemma-3-12b-it` returned `Error: Access denied. This repository requires
  approval.` Root cause: `HF_TOKEN` was set on the container at boot but
  RunPod doesn't propagate it into fresh ssh sessions; the env was empty.
- Patched `pod_runner.sh` to `source /etc/rp_environment` at the top so
  `hf` sees the token. Confirmed user `syvb` has Gemma access via API.
- Cleared partial downloads and `.cache/huggingface/hub`, restarted clean.

### 16:05 — Downloads complete
- 60 GB total: gemma-3-12b-it 23 G, nla-av 22 G, nla-ar 16 G. 81 GB free.

### 16:08 — Phase 1 false start #1
- `RuntimeError: could not find transformer-layer ModuleList on Gemma3ForConditionalGeneration`.
- transformers 5.6.0 wraps the multimodal model so the text-decoder
  ModuleList lives at `model.language_model.layers`, not the
  `language_model.layers` the script was trying. Patched
  `find_layers_module_list` with the new path.

### 16:12 — Phase 1 false start #2
- `RuntimeError: cuDNN Frontend error: [cudnn_frontend] Error: No valid execution plans built.`
- cuDNN's SDPA backend on driver 580 + cuDNN 9.x can't build a plan for
  Gemma-3's attention shape. Patched `cmd_extract` to call
  `torch.backends.cuda.enable_cudnn_sdp(False)` so PyTorch falls back to
  flash / mem-efficient SDPA backends. Both patches saved upstream in the
  local repo copy of `recon_loss_sweep.py`.

### 16:14–16:25 — Phase 1 actually runs
- 20,000 activations / 11.2 min, ~30 samples/s on Gemma-3-12B with
  batch_size=8, max_len=512. GPU mem 26 GB / 80, util 80–100%. 110 MB parquet.

### 16:26 — sglang launch fails on sgl-kernel
- `[sgl_kernel] CRITICAL: Could not load any common_ops library!` — the wheel
  expects libnvrtc.so.13 (CUDA 13) but the base image is CUDA 12.9.
- `pip install nvidia-cuda-nvrtc>=13` — already-satisfied, lib lives at
  `/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libnvrtc.so.13`.
- Patched `pod_runner.sh` to prepend that to `LD_LIBRARY_PATH` before
  launching sglang. `python -c "import sgl_kernel"` then succeeds.

### 16:34 — sglang `--disable-piecewise-cuda-graph`
- `Piecewise CUDA Graph failed with error: CUDA error: an illegal memory access`
  during warmup. sglang's own message says: pass
  `--disable-piecewise-cuda-graph`. Did so; sglang then warms up cleanly.

### 16:36 — sglang ready, AV smoke test
- Single AV decode through SGLang on a real activation works and the output
  is sensible English (not CJK soup). For a "My" token in a SQL Q&A context,
  AV emits "Technical Q&A forum structure with Russian/SQL query format
  ... Final token 'My' is part of the table name identifier 'dbo.MyTable'".
  Quality looks accurate.

### 16:38 — `nla_inference.py` patched for transformers 5.x
- `tokenizer.apply_chat_template(tokenize=True)` no longer returns a flat
  `list[int]` — it now returns a `BatchEncoding` whose first row is an
  `Encoding`, with int ids in `.ids`. Old code did
  `for i, tok in enumerate(ids)` and matched 0 injection tokens.
- Added `_flatten_chat_ids()` shim covering both old and new shapes; called
  in `load_nla_config` and `NLAClient._build_embeds`.

### 16:42 — fp16 storage destroys data → switching to fp32
- Smoke test surfaced `vec_norm = inf` on the second sample. Counted across
  the 20k dry-run extract: **64% of vectors have at least one +/-inf
  element**, because Gemma-3 layer-32 residual elements routinely exceed
  fp16's 65504 max. Storage as fp16 was silently corrupting most of the
  dataset. The mean *finite* vec_norm is ~58k.
- Switched extract to store activations as fp32 (column renamed
  `activation_fp16` → `activation`); decode reads either name for backward
  compat. fp32 ≈ 1.5 GB at 100k — fine on disk.
- Killed sglang to free GPU; relaunching after re-extract.

### 16:35–16:51 — Phase 1 re-extract (fp32)
- 20k activations / 11.2 min, 110 MB parquet (zstd compresses fp32 well).
  0 inf, 0 nan. Norms 25.9k–477k, median 72.8k.

### 16:53 — Decode crashes on transient httpx.ReadError
- After ~3 sec sglang dropped a connection mid-response under 16-way
  concurrency. Patched `cmd_decode` to add a 4-attempt exponential-backoff
  retry on `(ReadError, RemoteProtocolError, ConnectError, PoolTimeout)`,
  and bumped httpx pool limits to `4× concurrency`.

### 16:55–18:59 — Phase 2 decode (20k)
- Steady-state 2.7-2.8 samples/s; sglang gen throughput 500-700 tok/s on
  Gemma-3-12B with continuous batching. 124 minutes wall, no further errors.
- Output: 225 MB results parquet (full activation + recon vectors stored).

### 19:00 — Pull + destroy
- `scp` both parquets to `experiments/recon_loss_sweep/data/`.
- `DELETE /v1/pods/abfk12qdlxwyvp` → 404 confirms pod is gone.
- Pod ran for 3h 41m × $2.99/hr ≈ **$11.02**.

### 19:01 — Analysis
- Ran `analyze.py` against the results parquet. Auto-generated tables in
  `results_dryrun_20k.md`. Hand-written summary in `findings.md`.
- Headline: AV parse rate 99.99 %, mean cos 0.993, median cos 0.996. Worst
  reconstruction concentrates on sentence-initial caps + proper nouns at
  low L2-norm; best on common content words and a "sink-token" cluster at
  extreme norm.

### 19:23 — Deeper analysis (Gemma)
- Wrote `analyze_deep.py` (token classification, AV fallback detection,
  filter validation). Output in `deep_analysis_20k.md`. Hand-written
  synthesis in `deep_findings.md`.
- Discovered AV's two memorised fallback templates and quantified their
  impact (the "research paper" template alone explains 19/32 catastrophic
  failures; combined with a low-norm + cap-no-space filter catches 30/32).

## 2026-05-10 (Qwen run)

### 19:57 — Pod #2: nla-recon-loss-qwen
- New H100 SXM 80GB, US-NE-1, $2.99/hr, 100GB disk (less needed for Qwen).
- Pod ID `bz1vn3o000gp9k` on `216.243.220.227:10305`.
- Generalised `pod_runner.sh` to switch models via `MODEL_TAG=qwen7|gemma12`.

### 20:00–20:08 — Qwen download (39 GB)
- `Qwen/Qwen2.5-7B-Instruct` 15 G, AV 15 G, AR 11 G. Faster than Gemma run
  thanks to smaller total bytes.

### 20:08–20:14 — Phase 1 extract (Qwen, L20)
- 20k activations / 5.6 min, **60 samples/s** (2× Gemma). 102 MB parquet.
- Norms 74–162 — perfectly bounded "100–170" band per the README. 0 inf/nan.

### 20:14–22:16 — Phase 2 decode (Qwen)
- 20k decoded in 122 min, same 2.8 samples/s as Gemma. AV throughput
  improvement from smaller model is wiped out by the AR-step
  serialisation in the decode loop, plus per-request overhead in sglang.

### 22:16 — Pull + destroy
- Both parquets pulled to `data/`. `DELETE /v1/pods/bz1vn3o000gp9k` → 404.
- Pod ran 2h 19m × $2.99/hr ≈ **$6.93**.

### 22:18 — Qwen analysis
- Ran `analyze.py` and `analyze_deep.py` with `--norm-cutoff 200` (no
  sink cluster — Qwen norms cap at 162). Reports in
  `qwen_results_20k.md`, `qwen_deep_analysis_20k.md`.
- Comparative findings vs Gemma in `qwen_findings.md`.
- Headline: Qwen mean cos = 0.876 (vs Gemma's 0.993). Different failure
  *character* — uniformly mediocre vs mostly-perfect-with-catastrophes.
