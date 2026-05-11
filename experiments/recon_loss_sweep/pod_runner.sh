#!/usr/bin/env bash
# Pod-side helper for the recon_loss_sweep experiment. Lives at
# /workspace/pod_runner.sh on the H100. Single source of truth — re-invoke
# instead of pasting variations.
#
# Subcommands:
#   download      # parallel HF downloads of base + AV + AR
#   sizes         # disk usage of each downloaded model
#   extract N     # phase 1: extract N activations to data/activations.parquet
#   av-up         # launch sglang AV server in background
#   av-wait       # block until sglang health probe passes
#   av-down       # kill the sglang server
#   decode        # phase 2: AV→AR round trip
#   pack          # tar+zstd the data dir for download
#   nuke          # delete the data dir (in case we need to redo)
#
# All long-running commands write a fresh log file under /workspace/logs/.
set -euo pipefail

# Pull RunPod-provided env vars (HF_TOKEN, etc) — fresh ssh sessions don't get
# them otherwise.
[ -f /etc/rp_environment ] && source /etc/rp_environment
export HF_TOKEN

# sglang-kernel 0.4.2.post1 is built against CUDA 13 NVRTC, but the base image
# ships CUDA 12.9. The nvidia-cuda-nvrtc-cu13 wheel installed alongside sglang
# bundles libnvrtc.so.13 in its lib dir; expose it.
NVRTC13=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
[ -d "$NVRTC13" ] && export LD_LIBRARY_PATH="$NVRTC13${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ROOT=/workspace
DATA=$ROOT/data
LOGS=$ROOT/logs
MODELS=$ROOT/models
PORT=30000
SGLANG_PIDFILE=$ROOT/sglang.pid

mkdir -p "$DATA" "$LOGS" "$MODELS"

# Pick model by MODEL_TAG env (default: gemma12). Each tag fixes the base
# repo, the NLA pair, and the layer index the script extracts at.
case "${MODEL_TAG:-gemma12}" in
  gemma12)
    BASE_REPO=google/gemma-3-12b-it
    AV_REPO=kitft/nla-gemma3-12b-L32-av
    AR_REPO=kitft/nla-gemma3-12b-L32-ar
    LAYER=32
    EXTRACT_BATCH=8
    ;;
  qwen7)
    BASE_REPO=Qwen/Qwen2.5-7B-Instruct
    AV_REPO=kitft/nla-qwen2.5-7b-L20-av
    AR_REPO=kitft/nla-qwen2.5-7b-L20-ar
    LAYER=20
    EXTRACT_BATCH=16
    ;;
  *)
    echo "unknown MODEL_TAG=$MODEL_TAG (expected gemma12 | qwen7)" >&2
    exit 2
    ;;
esac
BASE_DIR=$MODELS/${MODEL_TAG}-base
AV_DIR=$MODELS/${MODEL_TAG}-av
AR_DIR=$MODELS/${MODEL_TAG}-ar

dl_one() {
  local repo=$1 dir=$2 logf=$3
  HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download "$repo" --local-dir "$dir" \
    > "$logf" 2>&1
}

cmd_download() {
  echo "[download] starting parallel downloads to $MODELS"
  dl_one "$BASE_REPO" "$BASE_DIR" "$LOGS/dl_base.log" &
  P1=$!
  dl_one "$AV_REPO"   "$AV_DIR"   "$LOGS/dl_av.log"   &
  P2=$!
  dl_one "$AR_REPO"   "$AR_DIR"   "$LOGS/dl_ar.log"   &
  P3=$!
  wait $P1 $P2 $P3
  echo "[download] done"
  cmd_sizes
}

cmd_sizes() {
  du -sh "$BASE_DIR" "$AV_DIR" "$AR_DIR" 2>/dev/null || true
  df -h /workspace
}

cmd_extract() {
  local n=${1:-20000}
  local out=$DATA/activations_${MODEL_TAG:-gemma12}_${n}.parquet
  local logf=$LOGS/extract_${MODEL_TAG:-gemma12}_${n}.log
  echo "[extract] tag=${MODEL_TAG:-gemma12} layer=$LAYER n=$n  out=$out  log=$logf"
  python /workspace/recon_loss_sweep.py extract \
    --base-model "$BASE_DIR" \
    --layer "$LAYER" \
    --n "$n" \
    --max-len 512 \
    --batch-size "$EXTRACT_BATCH" \
    --out "$out" \
    > "$logf" 2>&1
  echo "[extract] done. $(du -h "$out" | cut -f1) → $out"
}

cmd_av_up() {
  local logf=$LOGS/sglang.log
  if [ -f "$SGLANG_PIDFILE" ] && kill -0 "$(cat $SGLANG_PIDFILE)" 2>/dev/null; then
    echo "[av-up] already running pid=$(cat $SGLANG_PIDFILE)"
    return 0
  fi
  echo "[av-up] launching sglang on :$PORT  log=$logf"
  # mem-fraction-static 0.55 leaves room for AR (~16 GB) on the same H100.
  # context-length 512 — NLA prompts are short.
  SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1 \
    nohup python -m sglang.launch_server \
      --model-path "$AV_DIR" \
      --port "$PORT" \
      --disable-radix-cache \
      --disable-piecewise-cuda-graph \
      --mem-fraction-static 0.55 \
      --context-length 512 \
      --trust-remote-code \
      > "$logf" 2>&1 &
  echo $! > "$SGLANG_PIDFILE"
  echo "[av-up] pid=$(cat $SGLANG_PIDFILE)"
}

cmd_av_wait() {
  echo "[av-wait] waiting for /health on :$PORT"
  for _ in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "[av-wait] sglang ready"
      return 0
    fi
    sleep 5
  done
  echo "[av-wait] TIMED OUT after 20 min" >&2
  tail -50 "$LOGS/sglang.log" >&2
  return 1
}

cmd_av_down() {
  if [ -f "$SGLANG_PIDFILE" ]; then
    pid=$(cat $SGLANG_PIDFILE)
    echo "[av-down] killing pid=$pid"
    kill "$pid" 2>/dev/null || true
    sleep 3
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$SGLANG_PIDFILE"
  fi
  # Double-tap any orphan sglang processes (workers spawn children).
  pkill -f sglang.launch_server 2>/dev/null || true
  pkill -f sglang_router 2>/dev/null || true
  sleep 2
}

cmd_decode() {
  local n=${1:-20000}
  local tag=${MODEL_TAG:-gemma12}
  local act=$DATA/activations_${tag}_${n}.parquet
  local out=$DATA/results_${tag}_${n}.parquet
  local logf=$LOGS/decode_${tag}_${n}.log
  echo "[decode] tag=$tag n=$n  in=$act  out=$out  log=$logf"
  python /workspace/recon_loss_sweep.py decode \
    --activations "$act" \
    --av-checkpoint "$AV_DIR" \
    --ar-checkpoint "$AR_DIR" \
    --sglang-url "http://localhost:$PORT" \
    --ar-device cuda:0 \
    --av-concurrency 16 \
    --batch-size 64 \
    --out "$out" \
    > "$logf" 2>&1
  echo "[decode] done. $(du -h "$out" | cut -f1) → $out"
}

cmd_pack() {
  local tag=${1:-results}
  local arc=/workspace/${tag}.tar.zst
  echo "[pack] $arc"
  tar --use-compress-program='zstd -T0 -3' -cf "$arc" -C /workspace data logs
  ls -lh "$arc"
}

cmd_nuke() {
  rm -rf "$DATA"/*
  echo "[nuke] cleared $DATA"
}

cmd=${1:-}
shift || true
case "$cmd" in
  download)  cmd_download "$@";;
  sizes)     cmd_sizes "$@";;
  extract)   cmd_extract "$@";;
  av-up)     cmd_av_up "$@";;
  av-wait)   cmd_av_wait "$@";;
  av-down)   cmd_av_down "$@";;
  decode)    cmd_decode "$@";;
  pack)      cmd_pack "$@";;
  nuke)      cmd_nuke "$@";;
  *) echo "usage: $0 {download|sizes|extract N|av-up|av-wait|av-down|decode N|pack TAG|nuke}" >&2; exit 2;;
esac
