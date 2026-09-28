#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-MapTR}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the nuScenes root}"
CAN_BUS_ROOT="${CAN_BUS_ROOT:-${DATA_ROOT%/*}}"
SCENE_NAME="${SCENE_NAME:-scene-0061}"
WORK_DIR="${WORK_DIR:-${ROOT}/.tmp/maptrv2}"

MUSPACE_STRICT_PATCH=1 MUSPACE_DISPATCH_LOG=1 python "${ROOT}/model_examples/MapTRv2/scripts/train_step_musa.py" \
    --checkout "${CHECKOUT}" --data-root "${DATA_ROOT}" --can-bus-root "${CAN_BUS_ROOT}" \
    --scene-name "${SCENE_NAME}" --work-dir "${WORK_DIR}" --device musa:0 "$@"
