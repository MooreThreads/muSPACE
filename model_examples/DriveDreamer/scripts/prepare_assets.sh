#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="DriveDreamer"
RAW_ROOT=""
OUTPUT_ROOT=""
MODELS_DIR=""
DOWNLOAD_SD15=0

usage() {
    cat <<'USAGE'
Usage: prepare_assets.sh --raw-nuscenes ROOT --output-root ROOT --models-dir DIR [--checkout PATH] [--download-sd15]

Validates a licensed nuScenes mini download, converts it with DriveDreamer's official
converter, and optionally downloads the SD 1.5 and CLIP models into --models-dir.
The script never modifies the DriveDreamer checkout.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2 ;;
        --raw-nuscenes) RAW_ROOT="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --download-sd15) DOWNLOAD_SD15=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${RAW_ROOT}" && -n "${OUTPUT_ROOT}" && -n "${MODELS_DIR}" ]] || { usage >&2; exit 2; }
[[ -d "${RAW_ROOT}/v1.0-mini" && -d "${RAW_ROOT}/samples" && -d "${RAW_ROOT}/sweeps" && -d "${RAW_ROOT}/maps/expansion" ]] || {
    echo "--raw-nuscenes must contain v1.0-mini/, samples/, sweeps/, and maps/expansion/" >&2; exit 2;
}
[[ -f "${CHECKOUT}/dreamer-datasets/dd_scripts/converters/nuscenes_converter.py" ]] || {
    echo "invalid DriveDreamer checkout: ${CHECKOUT}" >&2; exit 2;
}

if [[ ${DOWNLOAD_SD15} -eq 1 ]]; then
    command -v huggingface-cli >/dev/null || { echo "huggingface-cli is required for --download-sd15" >&2; exit 2; }
    # Keep only one weight format.  Downloading the upstream .bin files as well
    # as safetensors nearly doubles storage without helping this FP32 recipe.
    huggingface-cli download runwayml/stable-diffusion-v1-5 \
        --local-dir "${MODELS_DIR}/runwayml/stable-diffusion-v1-5" \
        --include 'model_index.json' 'scheduler/*' 'tokenizer/*' 'text_encoder/*.json' \
        'text_encoder/*.safetensors' 'unet/*.json' 'unet/*.safetensors' \
        'vae/*.json' 'vae/*.safetensors'
    huggingface-cli download openai/clip-vit-large-patch14 \
        --local-dir "${MODELS_DIR}/openai/clip-vit-large-patch14" \
        --include '*.json' '*.txt' '*.safetensors'
fi
for required in vae unet scheduler; do
    [[ -d "${MODELS_DIR}/runwayml/stable-diffusion-v1-5/${required}" ]] || { echo "missing SD 1.5 ${required}/ in ${MODELS_DIR}" >&2; exit 2; }
done
[[ -d "${MODELS_DIR}/openai/clip-vit-large-patch14" ]] || { echo "missing CLIP model in ${MODELS_DIR}" >&2; exit 2; }

mkdir -p "${OUTPUT_ROOT}"
# DriveDreamer's upstream converter appends v1.0-mini to --data_root, then
# passes that nested path to NuScenes.  A normal nuScenes extraction keeps
# samples/, sweeps/, maps/, and v1.0-mini/ as siblings, so build a link-only
# compatibility view instead of changing the licensed raw dataset layout.
CONVERTER_ROOT="${OUTPUT_ROOT}/.converter_input"
mkdir -p "${CONVERTER_ROOT}/v1.0-mini"
ln -sfn "${RAW_ROOT}/v1.0-mini" "${CONVERTER_ROOT}/v1.0-mini/v1.0-mini"
ln -sfn "${RAW_ROOT}/samples" "${CONVERTER_ROOT}/v1.0-mini/samples"
ln -sfn "${RAW_ROOT}/sweeps" "${CONVERTER_ROOT}/v1.0-mini/sweeps"
ln -sfn "${RAW_ROOT}/maps" "${CONVERTER_ROOT}/v1.0-mini/maps"
PYTHONPATH="${CHECKOUT}/dreamer-datasets${PYTHONPATH:+:${PYTHONPATH}}" \
python "${CHECKOUT}/dreamer-datasets/dd_scripts/converters/nuscenes_converter.py" \
    --nusc_version v1.0-mini --data_root "${CONVERTER_ROOT}" --save_root "${OUTPUT_ROOT}" --mode cam_all
echo "DriveDreamer mini assets ready under ${OUTPUT_ROOT}"
