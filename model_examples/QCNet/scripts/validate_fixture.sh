#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-QCNet}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/.tmp/qcnet_train}"
DTYPE="${DTYPE:-float32}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkout) CHECKOUT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${DATA_ROOT}" ]] || { echo "--data-root is required" >&2; exit 2; }
[[ "${DTYPE}" == "float32" || "${DTYPE}" == "float16" ]] || { echo "invalid --dtype" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"
python -m mu_space migrate qcnet --checkout "${CHECKOUT}" >/dev/null
python "${ROOT}/model_examples/QCNet/scripts/train_step_musa.py" \
  --checkout "${CHECKOUT}" --data-root "${DATA_ROOT}" --dtype "${DTYPE}" \
  --output-dir "${OUTPUT_DIR}" | tee "${OUTPUT_DIR}/stdout.json"
python - "${OUTPUT_DIR}/result.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
errors = []
if payload.get("mode") != "real_subset": errors.append("mode is not real_subset")
if not payload.get("device", "").startswith("musa"): errors.append("not a MUSA run")
if payload.get("optimizer_step") is not True: errors.append("optimizer_step is not true")
if payload.get("finite_gradients") is not True: errors.append("gradients are not finite")
if payload.get("finite_parameters") is not True: errors.append("parameters are non-finite")
for key in ("batch", "initial_state", "trained_state"):
    if not Path(payload[key]).is_file(): errors.append(f"missing {key}")
if errors: raise SystemExit("QCNet validation failed: " + "; ".join(errors))
print(f"QCNet validation passed: {sys.argv[1]}")
PY
