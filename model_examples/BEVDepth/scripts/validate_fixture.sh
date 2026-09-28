#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-BEVDepth}"
DATA_ROOT="${DATA_ROOT:-}"
DTYPE="${DTYPE:-float32}"
DEVICE="${DEVICE:-musa}"

usage() {
    cat <<'USAGE'
Usage: model_examples/BEVDepth/scripts/validate_fixture.sh \
  --checkout PATH --data-root PATH --dtype float32|float16 [--device musa|cuda] \
  [--epochs N] [--max-hours H] [--max-steps N]
USAGE
}

SEED=2333
SCENE_NAME=scene-0001
NUM_WORKERS=0
EPOCHS=5
MAX_HOURS=8
MAX_STEPS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --scene-name) SCENE_NAME="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${DATA_ROOT}" ]]; then
    echo "--data-root or DATA_ROOT is required" >&2
    usage >&2
    exit 2
fi

case "${DTYPE}" in
    float32|float16) ;;
    *) echo "--dtype must be float32 or float16" >&2; exit 2 ;;
esac
case "${DEVICE}" in
    musa|cuda) ;;
    *) echo "--device must be musa or cuda" >&2; exit 2 ;;
esac

result=$(PYTHONPATH="${ROOT}:${CHECKOUT}:${ROOT}/model_examples/BEVDepth/scripts" \
  python "${ROOT}/model_examples/BEVDepth/scripts/train_step_musa.py" \
    --checkout "${CHECKOUT}" --data-root "${DATA_ROOT}" --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --scene-name "${SCENE_NAME}" --seed "${SEED}" --num-workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" --max-hours "${MAX_HOURS}" --max-steps "${MAX_STEPS}")
printf '%s\n' "${result}"
python - "${result}" <<'PY'
import json
import math
import sys

payload = json.loads(sys.argv[1].splitlines()[-1])
if not math.isfinite(payload.get("loss", float("nan"))):
    raise SystemExit("BEVDepth loss is not finite")
if not payload.get("finite_gradients") or not payload.get("optimizer_step"):
    raise SystemExit("BEVDepth single-step regression")
dispatch = payload.get("dispatch") or []
if not isinstance(dispatch, list) or {item.get("operator") for item in dispatch} != {"voxel_pooling_train", "deform_conv2d"}:
    raise SystemExit("BEVDepth did not execute both native operators")
if payload.get("device", "").startswith("musa"):
    if any(item.get("selected_backend") != "musa" or item.get("native_fallback") for item in dispatch):
        raise SystemExit("BEVDepth did not execute strict MUSA voxel pooling")
print(f"BEVDepth single-step {payload.get('device', 'unknown')} validation passed")
PY
