#!/usr/bin/env bash
# Pod-side helper for the recon_loss_sweep experiment. Lives at
# /workspace/pod_runner.sh on the H100. Single source of truth.
#
# Subcommands:
#   download | sizes | extract N | av-up | av-wait | av-down | decode N | nuke
#
# Pick model with MODEL_TAG env: gemma12 (default) or qwen7.
set -euo pipefail

# Pull RunPod-provided env (HF_TOKEN, etc).
[ -f /etc/rp_environment ] && source /etc/rp_environment
export HF_TOKEN

# sglang-kernel 0.4.2.post1 is built against CUDA 13 NVRTC; expose it.
NVRTC13=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
[ -d "$NVRTC13" ] && export LD_LIBRARY_PATH="$NVRTC13${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ROOT=/workspace
DATA=$ROOT/data
LOGS=$ROOT/logs
MODELS=$ROOT/models
PORT=30000
SGLANG_PIDFILE=$ROOT/sglang.pid

mkdir -p "$DATA" "$LOGS" "$MODELS"

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
    echo "unknown MODEL_TAG=$MODEL_TAG" >&2; exit 2;;
esac
BASE_DIR=$MODELS/${MODEL_TAG}-base
AV_DIR=$MODELS/${MODEL_TAG}-av
AR_DIR=$MODELS/${MODEL_TAG}-ar

dl_one() {
  local repo=$1 dir=$2 logf=$3
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$repo" --local-dir "$dir" \
    > "$logf" 2>&1
}

cmd_download() {
  echo "[download] starting parallel downloads to $MODELS"
  dl_one "$BASE_REPO" "$BASE_DIR" "$LOGS/dl_base.log" & P1=$!
  dl_one "$AV_REPO"   "$AV_DIR"   "$LOGS/dl_av.log"   & P2=$!
  dl_one "$AR_REPO"   "$AR_DIR"   "$LOGS/dl_ar.log"   & P3=$!
  wait $P1 $P2 $P3
  echo "[download] done"
  cmd_sizes
}

cmd_sizes() { du -sh "$BASE_DIR" "$AV_DIR" "$AR_DIR" 2>/dev/null || true; df -h /workspace; }

cmd_extract() {
  local n=${1:-20000}
  local seed=${SEED:-0}
  local sampling=${SAMPLING:-diverse_shards}
  local tag=${MODEL_TAG:-gemma12}
  local out=$DATA/activations_${tag}_${sampling}_seed${seed}_${n}.parquet
  local logf=$LOGS/extract_${tag}_${sampling}_seed${seed}_${n}.log
  echo "[extract] tag=$tag layer=$LAYER n=$n sampling=$sampling seed=$seed"
  echo "[extract] out=$out  log=$logf"
  python /workspace/recon_loss_sweep.py extract \
    --base-model "$BASE_DIR" --layer "$LAYER" \
    --n "$n" --max-len 512 --batch-size "$EXTRACT_BATCH" \
    --sampling "$sampling" --seed "$seed" \
    --out "$out" \
    > "$logf" 2>&1
  echo "[extract] done. $(du -h "$out" | cut -f1) → $out"
}

cmd_av_up() {
  local logf=$LOGS/sglang.log
  if [ -f "$SGLANG_PIDFILE" ] && kill -0 "$(cat $SGLANG_PIDFILE)" 2>/dev/null; then
    echo "[av-up] already running pid=$(cat $SGLANG_PIDFILE)"; return 0
  fi
  echo "[av-up] launching sglang on :$PORT  log=$logf"
  SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1 \
    nohup python -m sglang.launch_server \
      --model-path "$AV_DIR" --port "$PORT" \
      --disable-radix-cache --disable-piecewise-cuda-graph \
      --mem-fraction-static 0.55 --context-length 512 \
      --trust-remote-code \
      > "$logf" 2>&1 &
  echo $! > "$SGLANG_PIDFILE"
  echo "[av-up] pid=$(cat $SGLANG_PIDFILE)"
}

cmd_av_wait() {
  for _ in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo READY; return 0; }
    sleep 5
  done
  echo "TIMED OUT" >&2; tail -50 "$LOGS/sglang.log" >&2; return 1
}

cmd_av_down() {
  if [ -f "$SGLANG_PIDFILE" ]; then
    pid=$(cat $SGLANG_PIDFILE)
    kill "$pid" 2>/dev/null || true; sleep 3; kill -9 "$pid" 2>/dev/null || true
    rm -f "$SGLANG_PIDFILE"
  fi
  pkill -f sglang.launch_server 2>/dev/null || true
  pkill -f sglang_router 2>/dev/null || true
  sleep 2
}

cmd_decode() {
  local n=${1:-20000}
  local seed=${SEED:-0}
  local sampling=${SAMPLING:-diverse_shards}
  local tag=${MODEL_TAG:-gemma12}
  local act=$DATA/activations_${tag}_${sampling}_seed${seed}_${n}.parquet
  local out=$DATA/results_${tag}_${sampling}_seed${seed}_${n}.parquet
  local logf=$LOGS/decode_${tag}_${sampling}_seed${seed}_${n}.log
  echo "[decode] tag=$tag n=$n  in=$act  out=$out"
  python /workspace/recon_loss_sweep.py decode \
    --activations "$act" --av-checkpoint "$AV_DIR" --ar-checkpoint "$AR_DIR" \
    --sglang-url "http://localhost:$PORT" --ar-device cuda:0 \
    --av-concurrency 16 --batch-size 64 --out "$out" \
    > "$logf" 2>&1
  echo "[decode] done. $(du -h "$out" | cut -f1) → $out"
}

cmd_nuke() { rm -rf "$DATA"/*; }

cmd=${1:-}; shift || true
case "$cmd" in
  download) cmd_download "$@";;
  sizes)    cmd_sizes "$@";;
  extract)  cmd_extract "$@";;
  av-up)    cmd_av_up "$@";;
  av-wait)  cmd_av_wait "$@";;
  av-down)  cmd_av_down "$@";;
  decode)   cmd_decode "$@";;
  nuke)     cmd_nuke "$@";;
  *) echo "usage: $0 {download|sizes|extract N|av-up|av-wait|av-down|decode N|nuke}" >&2; exit 2;;
esac
