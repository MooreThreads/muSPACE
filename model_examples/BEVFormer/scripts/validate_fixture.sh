#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-BEVFormer}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/build/validation/bevformer_fixture}"
SEED="${SEED:-20260714}"
DEVICE="${DEVICE:-musa}"
STRICT=0

usage() {
    cat <<'USAGE'
Usage: model_examples/BEVFormer/scripts/validate_fixture.sh [--checkout PATH] [--output-dir PATH]
                                                            [--seed N] [--device musa|cuda] [--strict]

Runs one BEVFormer fixture optimizer step and validates the JSON result.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout)
            CHECKOUT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --strict)
            STRICT=1
            shift
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

case "${DEVICE}" in
    musa|cuda) ;;
    *)
        echo "--device must be musa or cuda" >&2
        exit 2
        ;;
esac

mkdir -p "${OUTPUT_DIR}"
mode_name="default"
if [[ "${STRICT}" -eq 1 ]]; then
    mode_name="strict"
fi
output="${OUTPUT_DIR}/${mode_name}.json"

cmd=(
    python "${ROOT}/model_examples/BEVFormer/scripts/train_step_musa.py"
    --checkout "${CHECKOUT}"
    --device "${DEVICE}"
    --mode fixture
    --seed "${SEED}"
)

if [[ "${STRICT}" -eq 1 ]]; then
    MUSPACE_STRICT_PATCH=1 "${cmd[@]}" | tee "${output}"
else
    "${cmd[@]}" | tee "${output}"
fi

python - "${output}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
errors = []
if payload.get("mode") != "fixture":
    errors.append("mode is not fixture")
if payload.get("finite_gradients") is not True:
    errors.append("finite_gradients is not true")
if payload.get("optimizer_step") is not True:
    errors.append("optimizer_step is not true")
loss = payload.get("loss")
if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
    errors.append("loss is missing or non-finite")
losses = payload.get("losses")
if not isinstance(losses, dict) or not losses:
    errors.append("losses is missing or empty")
if errors:
    raise SystemExit("BEVFormer fixture validation failed: " + "; ".join(errors))
print(f"BEVFormer fixture validation passed: {path}")
PY
