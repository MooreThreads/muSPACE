#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-BEVFusion}"
DEVICE="${DEVICE:-musa}"
BACKEND="${BACKEND:-musa}"
SEED="${SEED:-20260810}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

result=$(PYTHONPATH="${ROOT}" python "${ROOT}/model_examples/BEVFusion/scripts/train_step_musa.py" \
    --checkout "${CHECKOUT}" --device "${DEVICE}" --backend "${BACKEND}" --seed "${SEED}")
printf '%s\n' "${result}"
python - "${result}" <<'PY'
import json
import math
import sys

payload = json.loads(sys.argv[1].splitlines()[-1])
errors = []
if payload.get("optimizer_step") is not True:
    errors.append("optimizer_step is not true")
if payload.get("finite_gradients") is not True:
    errors.append("finite_gradients is not true")
if not math.isfinite(payload.get("loss", float("nan"))):
    errors.append("loss is not finite")
report = payload.get("focal_loss_dispatch") or {}
if report.get("selected_backend") != "musa":
    errors.append("sigmoid focal loss did not select the native MUSA backend")
if errors:
    raise SystemExit("BEVFusion fixture validation failed: " + "; ".join(errors))
print("BEVFusion fixture validation passed")
PY
