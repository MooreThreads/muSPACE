#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-Sparse4D}"
DATA_ROOT="${DATA_ROOT:-}"
OUT_DIR="${OUT_DIR:-}"
VERSION="${VERSION:-v1.0-trainval}"
MAX_SWEEPS="${MAX_SWEEPS:-10}"

usage() {
    cat <<'USAGE'
Usage: model_examples/Sparse4D/scripts/prepare_dataset.sh --checkout PATH --data-root PATH
                                                          [--out-dir PATH] [--version VERSION]
                                                          [--max-sweeps N]

Generates the Sparse4D training info files through the pinned checkout's own
tools/nuscenes_converter.py. The nuScenes root must contain the metadata, samples,
sweeps, and maps trees for the requested version.

--out-dir defaults to <checkout>/data/infos. --version accepts v1.0-trainval
(default), v1.0-mini, or v1.0-test.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --max-sweeps) MAX_SWEEPS="$2"; shift 2 ;;
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
    v1.0-trainval|v1.0-mini|v1.0-test) ;;
    *) echo "--version must be v1.0-trainval, v1.0-mini or v1.0-test" >&2; exit 2 ;;
esac

CONVERTER="${CHECKOUT}/tools/nuscenes_converter.py"
if [[ ! -f "${CONVERTER}" ]]; then
    echo "not a Sparse4D checkout: ${CHECKOUT}" >&2
    exit 2
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "dataset root does not exist: ${DATA_ROOT}" >&2
    exit 2
fi

if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${CHECKOUT}/data/infos"
fi

DATA_ROOT=$(cd "${DATA_ROOT}" && pwd)
mkdir -p "${OUT_DIR}"
OUT_DIR=$(cd "${OUT_DIR}" && pwd)

# The converter writes <info_prefix>_infos_{train,val,test}.pkl relative to its
# working directory, so pass an absolute prefix.
PYTHONPATH="${CHECKOUT}:${PYTHONPATH:-}" python "${CONVERTER}" \
    --root_path "${DATA_ROOT}" \
    --info_prefix "${OUT_DIR}/nuscenes" \
    --version "${VERSION}" \
    --max_sweeps "${MAX_SWEEPS}"

case "${VERSION}" in
    v1.0-test)
        EXPECTED=("${OUT_DIR}/nuscenes_infos_test.pkl")
        ;;
    *)
        EXPECTED=(
            "${OUT_DIR}/nuscenes_infos_train.pkl"
            "${OUT_DIR}/nuscenes_infos_val.pkl"
        )
        ;;
esac

for path in "${EXPECTED[@]}"; do
    if [[ ! -s "${path}" ]]; then
        echo "converter did not produce ${path}" >&2
        exit 1
    fi
    echo "wrote ${path}"
done
