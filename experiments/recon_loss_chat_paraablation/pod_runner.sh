#!/usr/bin/env bash
# Pod-side helper for the paragraph-ablation AR-only experiment.
# Single sub-command: download AR + run AR over input parquet.
#
# Reads HF_TOKEN from /etc/rp_environment (RunPod default).

set -euo pipefail

# Ensure HF_TOKEN is sourced into ssh sessions (RunPod doesn't propagate it).
if [[ -f /etc/rp_environment ]]; then
    # shellcheck disable=SC1091
    source /etc/rp_environment
fi

cd /workspace

cmd="${1:-help}"

case "$cmd" in
    install)
        echo "[pod] installing deps"
        python -m pip install -q "transformers>=5.6" "datasets>=4.0" "pyarrow>=20" \
            "numpy>=2" orjson safetensors pyyaml accelerate hf_transfer pandas
        echo "[pod] done"
        ;;

    download)
        echo "[pod] downloading AR checkpoint only"
        export HF_HUB_ENABLE_HF_TRANSFER=1
        hf download kitft/nla-gemma3-12b-L32-ar --local-dir nla-ar
        echo "[pod] done. disk:"
        df -h /workspace
        ;;

    run)
        echo "[pod] running AR experiment"
        python run_ar.py \
            --ar-checkpoint nla-ar \
            --input ar_input.parquet \
            --out ar_output.parquet \
            --ar-device cuda:0 \
            2>&1 | tee ar_run.log
        ;;

    all)
        "$0" install
        "$0" download
        "$0" run
        ;;

    help|*)
        echo "Usage: $0 {install|download|run|all}"
        ;;
esac
