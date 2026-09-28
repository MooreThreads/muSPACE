#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-BEVFusion}"
CONFIG="${CONFIG:-configs/nuscenes/seg/fusion-bev256d2-lss.yaml}"
DATA_ROOT="${DATA_ROOT:-}"
SCENE_NAME="${SCENE_NAME:-scene-0001}"
MAX_ITERS="${MAX_ITERS:-1}"
DEVICE="${DEVICE:-musa:0}"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}}"
ARTIFACT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout|--config|--data-root|--scene-name|--max-iters|--device|--work-dir|--artifact)
            case "$1" in
                --checkout) CHECKOUT="$2" ;;
                --config) CONFIG="$2" ;;
                --data-root) DATA_ROOT="$2" ;;
                --scene-name) SCENE_NAME="$2" ;;
                --max-iters) MAX_ITERS="$2" ;;
                --device) DEVICE="$2" ;;
                --work-dir) WORK_DIR="$2" ;;
                --artifact) ARTIFACT="$2" ;;
            esac
            shift 2 ;;
        -h|--help)
            echo "Usage: $0 --checkout PATH [--config PATH] [--data-root PATH] [--scene-name NAME] [--max-iters N] [--device musa:0] [--work-dir PATH] [--artifact PATH]"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${DATA_ROOT}" ]]; then DATA_ROOT="${CHECKOUT}/data/nuscenes"; fi
[[ -d "${DATA_ROOT}" ]] || { echo "dataset root does not exist: ${DATA_ROOT}" >&2; exit 2; }

MUSPACE_STRICT_PATCH=1 MUSPACE_SPCONV_FP32_TF32=0 \
python "${ROOT}/model_examples/BEVFusion/scripts/train_fusion_musa.py" \
    --checkout "${CHECKOUT}" --config "${CONFIG}" --data-root "${DATA_ROOT}" \
    --scene-name "${SCENE_NAME}" --max-iters "${MAX_ITERS}" --device "${DEVICE}" \
    --work-dir "${WORK_DIR}" ${ARTIFACT:+--artifact "${ARTIFACT}"}
