#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-SparseDrive}"
DATA_ROOT="${DATA_ROOT:-}"
INFOS="${INFOS:-}"
DEVICE="${DEVICE:-musa:0}"

usage() {
    cat <<'USAGE'
Usage: model_examples/SparseDrive/scripts/validate_fixture.sh \
  --checkout PATH --data-root PATH --infos PATH [--device musa:0] \
  [--scene-name NAME] [--seed N] [--max-iters N] [--work-dir PATH]

Runs the SparseDrive single-scene launch check and validates native dispatch.
The info file must be produced first by prepare_dataset.sh.
USAGE
}

SEED=20260826
SCENE_NAME=scene-0001
MAX_ITERS=1
WORK_DIR="${TMPDIR:-/tmp}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --infos) INFOS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --scene-name) SCENE_NAME="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --max-iters) MAX_ITERS="$2"; shift 2 ;;
        --work-dir) WORK_DIR="$2"; shift 2 ;;
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

case "${DEVICE}" in
    musa*) ;;
    *) echo "--device must be a MUSA device, such as musa:0" >&2; exit 2 ;;
esac

result=$(python "${ROOT}/model_examples/SparseDrive/scripts/train_step_musa.py" \
    --checkout "${CHECKOUT}" --config projects/configs/sparsedrive_small_stage1.py \
    --data-root "${DATA_ROOT}" --infos "${INFOS}" \
    --scene-name "${SCENE_NAME}" --max-iters "${MAX_ITERS}" --device "${DEVICE}" \
    --seed "${SEED}" --work-dir "${WORK_DIR}")
printf '%s\n' "${result}"
python - "${result}" <<'PY'
import json
import math
import sys

payload = json.loads(sys.argv[1].splitlines()[-1])
if not math.isfinite(payload.get("loss", float("nan"))):
    raise SystemExit("SparseDrive loss is not finite")
if not payload.get("finite_gradients") or not payload.get("optimizer_step"):
    raise SystemExit("SparseDrive single-step regression")
dispatch = payload.get("dispatch") or []
if not isinstance(dispatch, list) or {item.get("operator") for item in dispatch} != {
    "deformable_aggregation"
}:
    raise SystemExit("SparseDrive did not execute the deformable aggregation operator")
if payload.get("device", "").startswith("musa"):
    if any(
        item.get("selected_backend") != "musa"
        or item.get("native_fallback")
        or not item.get("native_available")
        for item in dispatch
    ):
        raise SystemExit("SparseDrive did not execute strict MUSA deformable aggregation")
print(
    f"SparseDrive single-step {payload.get('device', 'unknown')} validation passed "
    f"({payload.get('dispatch_calls', 0)} deformable aggregation calls)"
)
PY
