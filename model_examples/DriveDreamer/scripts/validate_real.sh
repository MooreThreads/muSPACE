#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${ROOT}/DriveDreamer"; DATA_ROOT=""; MODELS_DIR=""; INPUT_ARTIFACT=""; ARTIFACT=""
DEVICE=musa; SEED=20260825; SAMPLE_INDEX=0
usage() { echo "Usage: validate_real.sh --checkout PATH --data-root PATH --models-dir PATH --input-artifact PATH --artifact PATH [--device musa|cuda]"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2;;
        --data-root) DATA_ROOT="$2"; shift 2;;
        --models-dir) MODELS_DIR="$2"; shift 2;;
        --input-artifact) INPUT_ARTIFACT="$2"; shift 2;;
        --artifact) ARTIFACT="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --sample-index) SAMPLE_INDEX="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
    esac
done
[[ -n "${DATA_ROOT}" && -n "${MODELS_DIR}" && -n "${INPUT_ARTIFACT}" && -n "${ARTIFACT}" ]] || { usage >&2; exit 2; }
python "${ROOT}/model_examples/DriveDreamer/scripts/train_step_musa.py" --checkout "${CHECKOUT}" --data-root "${DATA_ROOT}" --models-dir "${MODELS_DIR}" --input-artifact "${INPUT_ARTIFACT}" --artifact "${ARTIFACT}" --device "${DEVICE}" --seed "${SEED}" --sample-index "${SAMPLE_INDEX}"
python - "${ARTIFACT}" <<'PY'
import math
import sys
import torch
result = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
loss = result["payload"]["loss"]
if not math.isfinite(float(loss)):
    raise SystemExit("DriveDreamer real validation failed: non-finite loss")
before = result["stages"]["tracked_parameter_before"]
after = result["stages"]["tracked_parameter_after"]
if torch.equal(before, after):
    raise SystemExit("DriveDreamer real validation failed: optimizer did not update")
print(f"DriveDreamer real validation passed: loss={float(loss):.8g} artifact={sys.argv[1]}")
PY
