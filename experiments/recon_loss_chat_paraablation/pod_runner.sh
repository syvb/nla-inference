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

    run-many)
        # Run AR over a list of (input → output) pairs. Pass pairs as
        # extra args: in1 out1 in2 out2 ...  Stream all output into one
        # combined log + per-pair logs.
        shift
        if (( $# % 2 != 0 )); then
            echo "run-many requires even number of args: in1 out1 in2 out2 ..." >&2
            exit 2
        fi
        : > ar_run_many.log
        while (( $# >= 2 )); do
            inp=$1; outp=$2; shift 2
            echo "[pod] AR: $inp -> $outp" | tee -a ar_run_many.log
            python run_ar.py \
                --ar-checkpoint nla-ar \
                --input "$inp" \
                --out "$outp" \
                --ar-device cuda:0 \
                2>&1 | tee -a ar_run_many.log
        done
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
