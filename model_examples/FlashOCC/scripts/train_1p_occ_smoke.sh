#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-FlashOCC}"
CONFIG="${CONFIG:-projects/configs/flashocc/flashocc-r50.py}"
DATA_ROOT="${DATA_ROOT:-}"
SCENE_NAME="${SCENE_NAME:-scene-0001}"
MAX_ITERS="${MAX_ITERS:-1}"
DEVICE="${DEVICE:-musa:0}"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}}"

usage() {
    cat <<'USAGE'
Usage: model_examples/FlashOCC/scripts/train_1p_occ_smoke.sh [--checkout PATH] [--config PATH]
                                                            [--data-root PATH] [--scene-name NAME]
                                                            [--max-iters N] [--device musa:0]
                                                            [--work-dir PATH]

Runs one FlashOcc occupancy training iteration on a single card as a launch check.
--data-root defaults to <checkout>/data/nuscenes.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout)
            CHECKOUT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --scene-name)
            SCENE_NAME="$2"
            shift 2
            ;;
        --max-iters)
            MAX_ITERS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${DATA_ROOT}" ]]; then
    DATA_ROOT="${CHECKOUT}/data/nuscenes"
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "dataset root does not exist: ${DATA_ROOT}" >&2
    exit 2
fi

MUSPACE_STRICT_PATCH="${MUSPACE_STRICT_PATCH:-1}" \
python "${ROOT}/model_examples/FlashOCC/scripts/train_step_musa.py" \
    --checkout "${CHECKOUT}" \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --scene-name "${SCENE_NAME}" \
    --max-iters "${MAX_ITERS}" \
    --device "${DEVICE}" \
    --work-dir "${WORK_DIR}"
