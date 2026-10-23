#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-SparseDrive}"
CONFIG="${CONFIG:-projects/configs/sparsedrive_small_stage1.py}"
DATA_ROOT="${DATA_ROOT:-}"
INFOS="${INFOS:-data/infos/nuscenes_infos_train.pkl}"
SCENE_NAME="${SCENE_NAME:-scene-0001}"
MAX_ITERS="${MAX_ITERS:-1}"
DEVICE="${DEVICE:-musa:0}"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}}"

usage() {
    cat <<'USAGE'
Usage: model_examples/SparseDrive/scripts/train_1p_sparsedrive_smoke.sh [--checkout PATH]
                                                                       [--config PATH]
                                                                       [--data-root PATH]
                                                                       [--infos PATH]
                                                                       [--scene-name NAME]
                                                                       [--max-iters N]
                                                                       [--device musa:0]
                                                                       [--work-dir PATH]

Runs one SparseDrive training iteration on a single card as a launch check.
--data-root defaults to <checkout>/data/nuscenes. --infos defaults to
<checkout>/data/infos/nuscenes_infos_train.pkl and must be produced first by
model_examples/SparseDrive/scripts/prepare_dataset.sh.
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
        --infos)
            INFOS="$2"
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

if [[ "${INFOS}" != /* ]]; then
    INFOS="${CHECKOUT}/${INFOS}"
fi

if [[ ! -f "${INFOS}" ]]; then
    echo "training info file does not exist: ${INFOS}" >&2
    echo "run model_examples/SparseDrive/scripts/prepare_dataset.sh first" >&2
    exit 2
fi

MUSPACE_STRICT_PATCH="${MUSPACE_STRICT_PATCH:-1}" \
python "${ROOT}/model_examples/SparseDrive/scripts/train_step_musa.py" \
    --checkout "${CHECKOUT}" \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --infos "${INFOS}" \
    --scene-name "${SCENE_NAME}" \
    --max-iters "${MAX_ITERS}" \
    --device "${DEVICE}" \
    --work-dir "${WORK_DIR}"
