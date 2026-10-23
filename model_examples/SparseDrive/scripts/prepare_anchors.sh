#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-SparseDrive}"
INFOS="${INFOS:-}"
OUT_DIR="${OUT_DIR:-}"

usage() {
    cat <<'USAGE'
Usage: model_examples/SparseDrive/scripts/prepare_anchors.sh --checkout PATH
                                                             [--infos PATH] [--out-dir PATH]

Generates the K-means anchors the SparseDrive configs load:

  data/kmeans/kmeans_det_900.npy
  data/kmeans/kmeans_map_100.npy
  data/kmeans/kmeans_motion_6.npy
  data/kmeans/kmeans_plan_6.npy

The upstream tools run from the checkout root and hardcode the relative data/infos and
data/kmeans paths. Run prepare_dataset.sh first: the anchors are clustered from the
generated training info file.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2 ;;
        --infos) INFOS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "${CHECKOUT}/tools/kmeans/kmeans_det.py" ]]; then
    echo "not a SparseDrive checkout: ${CHECKOUT}" >&2
    exit 2
fi

if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${CHECKOUT}/data/kmeans"
fi

if [[ -z "${INFOS}" ]]; then
    INFOS="${CHECKOUT}/data/infos/nuscenes_infos_train.pkl"
elif [[ "${INFOS}" != /* ]]; then
    # Resolve relative paths against the checkout, like train_step_musa.py does.
    INFOS="${CHECKOUT}/${INFOS}"
fi

if [[ ! -s "${INFOS}" ]]; then
    echo "training info file does not exist: ${INFOS}" >&2
    echo "run model_examples/SparseDrive/scripts/prepare_dataset.sh first" >&2
    exit 2
fi

pushd "${CHECKOUT}" >/dev/null
trap 'popd >/dev/null' EXIT

# Point the hardcoded relative paths at the requested locations.
INFOS_DIR=$(cd "$(dirname "${INFOS}")" && pwd)
if [[ "${INFOS_DIR}/$(basename "${INFOS}")" != "$(pwd)/data/infos/nuscenes_infos_train.pkl" ]]; then
    mkdir -p data/infos
    ln -sfn "${INFOS}" data/infos/nuscenes_infos_train.pkl
fi

mkdir -p "${OUT_DIR}"
OUT_DIR_ABS=$(cd "${OUT_DIR}" && pwd)
if [[ "${OUT_DIR_ABS}" != "$(pwd)/data/kmeans" ]]; then
    mkdir -p data
    ln -sfn "${OUT_DIR_ABS}" data/kmeans
fi

for script in kmeans_det kmeans_map kmeans_motion kmeans_plan; do
    if ! PYTHONPATH="${CHECKOUT}:${PYTHONPATH:-}" python "tools/kmeans/${script}.py"; then
        echo "${script}.py failed while clustering anchors" >&2
        if [[ "${script}" == "kmeans_plan" ]]; then
            cat >&2 <<'HINT'
kmeans_plan.py clusters ego trajectories per navigation command and needs frames for all
three commands. A split that lacks one of them cannot produce plan anchors; the nuScenes
mini split is such a case, so generate kmeans_plan_6.npy from the v1.0-trainval infos.
HINT
        fi
        exit 1
    fi
done

ANCHORS=(
    data/kmeans/kmeans_det_900.npy
    data/kmeans/kmeans_map_100.npy
    data/kmeans/kmeans_motion_6.npy
    data/kmeans/kmeans_plan_6.npy
)

for path in "${ANCHORS[@]}"; do
    if [[ ! -s "${path}" ]]; then
        echo "anchor generation did not produce ${path}" >&2
        exit 1
    fi
    echo "wrote ${CHECKOUT}/${path}"
done
