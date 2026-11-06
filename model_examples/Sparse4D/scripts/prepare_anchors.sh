#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-Sparse4D}"
INFOS="${INFOS:-}"
OUT_DIR="${OUT_DIR:-}"
OUTPUT_NAME="${OUTPUT_NAME:-nuscenes_kmeans900.npy}"
NUM_ANCHOR="${NUM_ANCHOR:-900}"
USE_PATCHER=1

usage() {
    cat <<'USAGE'
Usage: model_examples/Sparse4D/scripts/prepare_anchors.sh --checkout PATH
                                                           [--infos PATH] [--out-dir PATH]
                                                           [--output-name NAME]
                                                           [--num-anchor N] [--no-patch]

Generates the K-means anchor file the Sparse4D configs load through InstanceBank.

--infos defaults to <checkout>/data/infos/nuscenes_infos_train.pkl and must be
produced by prepare_dataset.sh first. --out-dir defaults to <checkout>/data/anchors.
The default output name is nuscenes_kmeans900.npy; the upstream tool's own default
has a leading underscore, so the name is always passed explicitly.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2 ;;
        --infos) INFOS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        --num-anchor) NUM_ANCHOR="$2"; shift 2 ;;
        --no-patch) USE_PATCHER=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

ANCHOR_TOOL="${CHECKOUT}/tools/anchor_generator.py"
if [[ ! -f "${ANCHOR_TOOL}" ]]; then
    echo "not a Sparse4D checkout: ${CHECKOUT}" >&2
    exit 2
fi

if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${CHECKOUT}/data/anchors"
fi

if [[ -z "${INFOS}" ]]; then
    INFOS="${CHECKOUT}/data/infos/nuscenes_infos_train.pkl"
elif [[ "${INFOS}" != /* ]]; then
    INFOS="${CHECKOUT}/${INFOS}"
fi

if [[ ! -s "${INFOS}" ]]; then
    echo "training info file does not exist: ${INFOS}" >&2
    echo "run model_examples/Sparse4D/scripts/prepare_dataset.sh first" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"
OUT_DIR=$(cd "${OUT_DIR}" && pwd)
OUTPUT_PATH="${OUT_DIR}/${OUTPUT_NAME}"

# anchor_generator.py imports projects.mmdet3d_plugin.core.box3d, which triggers the
# plugin package import. Route it through the runtime patcher so the unbuilt
# deformable_aggregation_ext does not block the import.
if [[ "${USE_PATCHER}" == "1" ]]; then
    python "${ROOT}/model_examples/Sparse4D/scripts/run_with_patcher.py" \
        "${CHECKOUT}" "${ANCHOR_TOOL}" \
        --ann_file "${INFOS}" \
        --num_anchor "${NUM_ANCHOR}" \
        --output_file_name "${OUTPUT_PATH}"
else
    PYTHONPATH="${CHECKOUT}:${PYTHONPATH:-}" python "${ANCHOR_TOOL}" \
        --ann_file "${INFOS}" \
        --num_anchor "${NUM_ANCHOR}" \
        --output_file_name "${OUTPUT_PATH}"
fi

if [[ ! -s "${OUTPUT_PATH}" ]]; then
    echo "anchor generation did not produce ${OUTPUT_PATH}" >&2
    exit 1
fi

echo "wrote ${OUTPUT_PATH}"
