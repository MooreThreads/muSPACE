#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${ROOT}/DriveDreamer"; DATA_ROOT=""; MODELS_DIR=""; OUTPUT_DIR=""
EPOCHS=1; GPU_ID=0; SEED=20260825; RESUME=0
usage() { echo "Usage: train_full_musa.sh --checkout PATH --data-root PATH --models-dir PATH --output-dir PATH [--epochs N] [--gpu-id N] [--seed N] [--resume]"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2;;
        --data-root) DATA_ROOT="$2"; shift 2;;
        --models-dir) MODELS_DIR="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --gpu-id) GPU_ID="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --resume) RESUME=1; shift;;
        -h|--help) usage; exit 0;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
    esac
done
[[ -n "${DATA_ROOT}" && -n "${MODELS_DIR}" && -n "${OUTPUT_DIR}" ]] || { usage >&2; exit 2; }
cmd=(python "${ROOT}/model_examples/DriveDreamer/scripts/train_full_musa.py" --checkout "${CHECKOUT}" --data-root "${DATA_ROOT}" --models-dir "${MODELS_DIR}" --output-dir "${OUTPUT_DIR}" --epochs "${EPOCHS}" --gpu-id "${GPU_ID}" --seed "${SEED}")
[[ ${RESUME} -eq 1 ]] && cmd+=(--resume)
MUSPACE_STRICT_PATCH=1 "${cmd[@]}"
