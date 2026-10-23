#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-SparseDrive}"
DATA_ROOT="${DATA_ROOT:-}"
OUT_DIR="${OUT_DIR:-}"
VERSION="${VERSION:-v1.0}"
MAX_SWEEPS="${MAX_SWEEPS:-10}"
WORKERS="${WORKERS:-4}"

usage() {
    cat <<'USAGE'
Usage: model_examples/SparseDrive/scripts/prepare_dataset.sh --checkout PATH --data-root PATH
                                                             [--out-dir PATH] [--version v1.0]
                                                             [--max-sweeps N] [--workers N]

Generates the SparseDrive training info files through the pinned checkout's own
converter (tools/data_converter/nuscenes_converter.py). The nuScenes root must contain
the trainval metadata, samples, sweeps, maps, and can_bus trees.

--out-dir defaults to <checkout>/data/infos, which is where the model configs look for
nuscenes_infos_train.pkl. --version v1.0 writes the trainval info files; --version
v1.0-mini writes them below <out-dir>/mini and is only useful for a fast sanity check.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --max-sweeps) MAX_SWEEPS="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${DATA_ROOT}" ]]; then
    echo "--data-root or DATA_ROOT is required" >&2
    usage >&2
    exit 2
fi

case "${VERSION}" in
    v1.0|v1.0-mini) ;;
    *) echo "--version must be v1.0 or v1.0-mini" >&2; exit 2 ;;
esac

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "dataset root does not exist: ${DATA_ROOT}" >&2
    exit 2
fi

CONVERTER="${CHECKOUT}/tools/data_converter/nuscenes_converter.py"
if [[ ! -f "${CONVERTER}" ]]; then
    echo "not a SparseDrive checkout: ${CHECKOUT}" >&2
    exit 2
fi

if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${CHECKOUT}/data/infos"
fi

DATA_ROOT=$(cd "${DATA_ROOT}" && pwd)
mkdir -p "${OUT_DIR}"
OUT_DIR=$(cd "${OUT_DIR}" && pwd)

# The converter resolves the nuScenes metadata, the map expansion, and can_bus from
# the paths it is given, so it can run from any directory.
PYTHONPATH="${CHECKOUT}:${PYTHONPATH:-}" python "${CONVERTER}" nuscenes \
    --root-path "${DATA_ROOT}" \
    --canbus "${DATA_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --extra-tag nuscenes \
    --version "${VERSION}" \
    --max-sweeps "${MAX_SWEEPS}" \
    --workers "${WORKERS}"

if [[ "${VERSION}" == "v1.0-mini" ]]; then
    EXPECTED=(
        "${OUT_DIR}/mini/nuscenes_infos_train.pkl"
        "${OUT_DIR}/mini/nuscenes_infos_val.pkl"
    )
else
    EXPECTED=(
        "${OUT_DIR}/nuscenes_infos_train.pkl"
        "${OUT_DIR}/nuscenes_infos_val.pkl"
    )
fi

for path in "${EXPECTED[@]}"; do
    if [[ ! -s "${path}" ]]; then
        echo "converter did not produce ${path}" >&2
        exit 1
    fi
    echo "wrote ${path}"
done
