#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-Sparse4D}"
DATA_ROOT="${DATA_ROOT:-}"
INFOS="${INFOS:-}"
ANCHORS="${ANCHORS:-}"
DEVICE="${DEVICE:-musa:0}"
STRICT=1

usage() {
    cat <<'USAGE'
Usage: model_examples/Sparse4D/scripts/validate_fixture.sh --checkout PATH --data-root PATH
                                                           [--infos PATH] [--anchors PATH]
                                                           [--device musa:0] [--config PATH]
                                                           [--scene-name NAME] [--seed N]
                                                           [--max-iters N] [--work-dir PATH]
                                                           [--strict|--no-strict]

Runs the Sparse4D single-scene launch check and validates native dispatch.
--infos and --anchors default to <checkout>/data/infos/nuscenes_infos_train.pkl and
<checkout>/data/anchors/nuscenes_kmeans900.npy; produce them first with
prepare_dataset.sh and prepare_anchors.sh. --strict is the default and requires
every deformable aggregation dispatch to use the native MUSA kernel without fallback.
USAGE
}

SEED=20261026
SCENE_NAME=scene-0001
MAX_ITERS=1
WORK_DIR="${TMPDIR:-/tmp}"
CONFIG=projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --infos) INFOS="$2"; shift 2 ;;
        --anchors) ANCHORS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --scene-name) SCENE_NAME="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --max-iters) MAX_ITERS="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
        --strict) STRICT=1; shift ;;
        --no-strict) STRICT=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${DATA_ROOT}" ]]; then
    echo "--data-root or DATA_ROOT is required" >&2
    usage >&2
    exit 2
fi

if [[ -z "${INFOS}" ]]; then
    INFOS="${CHECKOUT}/data/infos/nuscenes_infos_train.pkl"
fi
if [[ -z "${ANCHORS}" ]]; then
    ANCHORS="${CHECKOUT}/data/anchors/nuscenes_kmeans900.npy"
fi

case "${DEVICE}" in
    musa*) ;;
    *) echo "--device must be a MUSA device, such as musa:0" >&2; exit 2 ;;
esac

BACKEND=auto
if [[ "${STRICT}" == "1" ]]; then
    BACKEND=strict
fi

result=$(python "${ROOT}/model_examples/Sparse4D/scripts/train_step_musa.py" \
    --mode fixture --checkout "${CHECKOUT}" --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" --infos "${INFOS}" --anchors "${ANCHORS}" \
    --scene-name "${SCENE_NAME}" --max-iters "${MAX_ITERS}" --device "${DEVICE}" \
    --seed "${SEED}" --work-dir "${WORK_DIR}" --backend "${BACKEND}")
printf '%s\n' "${result}"
python - "${result}" "${STRICT}" <<'PY'
import json
import math
import sys

payload = json.loads(sys.argv[1].splitlines()[-1])
strict = sys.argv[2] == "1"

if not math.isfinite(payload.get("loss", float("nan"))):
    raise SystemExit("Sparse4D loss is not finite")
if not payload.get("finite_gradients") or not payload.get("optimizer_step"):
    raise SystemExit("Sparse4D single-step regression")
if payload.get("selected_parameter_before_sha256") == payload.get(
    "selected_parameter_after_sha256"
):
    raise SystemExit("Sparse4D did not update the selected parameter")
dispatch = payload.get("dispatch") or []
if not isinstance(dispatch, list) or {item.get("operator") for item in dispatch} != {
    "deformable_aggregation"
}:
    raise SystemExit("Sparse4D did not execute the deformable aggregation operator")
if strict:
    if payload.get("upstream_revision") != "4ec96edf0e7d9e80fbd75ef22e1d51a937d3a750":
        raise SystemExit("Sparse4D upstream revision does not match the pinned commit")
    if any(
        item.get("selected_backend") != "musa"
        or item.get("native_fallback")
        or not item.get("native_available")
        for item in dispatch
    ):
        raise SystemExit("Sparse4D did not execute strict MUSA deformable aggregation")
print(
    f"Sparse4D single-step {payload.get('device', 'unknown')} validation passed "
    f"({payload.get('dispatch_calls', 0)} deformable aggregation calls)"
)
PY
