#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-OpenVLA}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/.tmp/openvla_fixture}"
SEED="${SEED:-20260814}"
DEVICE="${DEVICE:-musa}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "${DEVICE}" in
    musa|musa:0) ;;
    *) echo "--device must be musa or musa:0" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_DIR}"
output="${OUTPUT_DIR}/default.json"
python -m mu_space migrate openvla --checkout "${CHECKOUT}" >/dev/null
python "${ROOT}/model_examples/OpenVLA/scripts/train_step_musa.py" \
    --checkout "${CHECKOUT}" --device "${DEVICE}" --mode fixture --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}" | tee "${output}"

python - "${output}" <<'PY'
import json
import math
import sys

from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
errors = []
if payload.get("mode") != "fixture":
    errors.append("mode is not fixture")
if payload.get("device") != "musa:0":
    errors.append("fixture did not run on musa:0")
if payload.get("optimizer_step") is not True:
    errors.append("optimizer_step is not true")
if payload.get("finite_gradients") is not True:
    errors.append("finite_gradients is not true")
if not all(math.isfinite(value) for value in payload.get("losses", [])):
    errors.append("losses are not finite")
if errors:
    raise SystemExit("OpenVLA fixture validation failed: " + "; ".join(errors))
print(f"OpenVLA fixture validation passed: {sys.argv[1]}")
PY
