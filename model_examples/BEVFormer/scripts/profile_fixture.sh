#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-BEVFormer}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/build/profiling/bevformer_fixture}"
SEED="${SEED:-20260714}"
DEVICE="${DEVICE:-musa}"
STRICT=0
TRACE=0
SAMPLE_ELEMENTS="${SAMPLE_ELEMENTS:-512}"

usage() {
    cat <<'USAGE'
Usage: model_examples/BEVFormer/scripts/profile_fixture.sh [--checkout PATH] [--output-dir PATH]
                                                           [--seed N] [--device musa|cuda]
                                                           [--strict] [--trace]

Runs BEVFormer fixture profiling diagnostics. Output is for triage, not a
release performance claim.
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
        --sample-elements)
            SAMPLE_ELEMENTS="$2"
            shift 2
            ;;
        --strict)
            STRICT=1
            shift
            ;;
        --trace)
            TRACE=1
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

profile_json="${OUTPUT_DIR}/${mode_name}_train_step.json"
dispatch_log="${OUTPUT_DIR}/${mode_name}_dispatch.log"
trace_json="${OUTPUT_DIR}/${mode_name}_trace.json"
summary_json="${OUTPUT_DIR}/${mode_name}_summary.json"

cmd=(
    python "${ROOT}/model_examples/BEVFormer/scripts/train_step_musa.py"
    --checkout "${CHECKOUT}"
    --device "${DEVICE}"
    --mode fixture
    --seed "${SEED}"
)

if [[ "${STRICT}" -eq 1 ]]; then
    MUSPACE_STRICT_PATCH=1 MUSPACE_DISPATCH_LOG=1 "${cmd[@]}" \
        > "${profile_json}" 2> "${dispatch_log}"
else
    MUSPACE_DISPATCH_LOG=1 "${cmd[@]}" > "${profile_json}" 2> "${dispatch_log}"
fi

if [[ "${TRACE}" -eq 1 ]]; then
    trace_cmd=(
        python "${ROOT}/model_examples/BEVFormer/scripts/trace_bevformer_musa.py"
        --checkout "${CHECKOUT}"
        --device "${DEVICE}"
        --mode fixture
        --seed "${SEED}"
        --sample-elements "${SAMPLE_ELEMENTS}"
        --output "${trace_json}"
    )
    if [[ "${STRICT}" -eq 1 ]]; then
        MUSPACE_STRICT_PATCH=1 "${trace_cmd[@]}"
    else
        "${trace_cmd[@]}"
    fi
fi

python - "${profile_json}" "${dispatch_log}" "${trace_json}" "${summary_json}" "${mode_name}" <<'PY'
import json
import sys
from pathlib import Path

profile_path = Path(sys.argv[1])
dispatch_path = Path(sys.argv[2])
trace_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
mode_name = sys.argv[5]
profile = json.loads(profile_path.read_text(encoding="utf-8"))
dispatch_lines = [
    line for line in dispatch_path.read_text(encoding="utf-8").splitlines()
    if line.startswith("muSPACE dispatch: ")
]
summary = {
    "profile_mode": mode_name,
    "profile_json": str(profile_path),
    "dispatch_log": str(dispatch_path),
    "trace_json": str(trace_path) if trace_path.is_file() else None,
    "device": profile.get("device"),
    "seed": profile.get("seed"),
    "elapsed_seconds": profile.get("elapsed_seconds"),
    "peak_memory_bytes": profile.get("peak_memory_bytes"),
    "finite_gradients": profile.get("finite_gradients"),
    "optimizer_step": profile.get("optimizer_step"),
    "loss": profile.get("loss"),
    "dispatch_records": len(dispatch_lines),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
