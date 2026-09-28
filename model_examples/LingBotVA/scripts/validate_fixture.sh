#!/usr/bin/env bash
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-LingBotVA}"
MODE="${MODE:-smoke}"
DEVICE="${DEVICE:-musa:0}"
ATTN_MODE="${ATTN_MODE:-flex}"
SEED="${SEED:-20260906}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/.tmp/lingbotva_fixture}"
DATASET_PATH="${DATASET_PATH:-}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-}"
NGPU="${NGPU:-1}"
NUM_STEPS="${NUM_STEPS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
IFNAME="${IFNAME:-}"

usage() {
    cat <<'USAGE'
Usage: model_examples/LingBotVA/scripts/validate_fixture.sh [options]

  --checkout PATH              Pinned LingBot-VA checkout (default LingBotVA)
  --mode smoke|train           smoke is dataset-free, train needs data and weights
  --device musa:0              Accelerator to validate
  --attn-mode flex|torch|flashattn
  --output-dir PATH            Artifact directory (default <repo>/.tmp/lingbotva_fixture)
  --seed N
  --dataset-path PATH          Required for --mode train
  --pretrained-model-path PATH Required for --mode train
  --ngpu N                     Cards for --mode train (default 1)
  --num-steps N                Optimizer steps for --mode train (default 1)
  --save-interval N            Checkpoint interval for --mode train (default 200)
  --ifname IFACE               MCCL/NCCL socket interface for multi-card runs

Dataset-free smoke (any MUSA stack):

  model_examples/LingBotVA/scripts/validate_fixture.sh \
    --checkout .tmp/LingBotVA --output-dir .tmp/lingbotva_fixture

Real-data validation (prepared LeRobot-Latent dataset and lingbot-va-base):

  model_examples/LingBotVA/scripts/validate_fixture.sh \
    --checkout .tmp/LingBotVA --mode train --ngpu 8 --num-steps 3 \
    --dataset-path <LEROBOT_LATENT_ROOT> --pretrained-model-path <LINGBOT_VA_BASE>
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) CHECKOUT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --attn-mode) ATTN_MODE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --pretrained-model-path) PRETRAINED_MODEL_PATH="$2"; shift 2 ;;
        --ngpu) NGPU="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
        --ifname) IFNAME="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${MODE}" in
    smoke|train) ;;
    *) echo "--mode must be smoke or train" >&2; exit 2 ;;
esac
case "${DEVICE}" in
    musa|musa:0|cuda|cuda:0) ;;
    *) echo "--device must be musa, musa:0, cuda or cuda:0" >&2; exit 2 ;;
esac
case "${ATTN_MODE}" in
    flex|torch|flashattn) ;;
    *) echo "--attn-mode must be flex, torch or flashattn" >&2; exit 2 ;;
esac
if [[ "${MODE}" == "train" ]]; then
    [[ -n "${DATASET_PATH}" ]] || { echo "--dataset-path is required for --mode train" >&2; exit 2; }
    [[ -n "${PRETRAINED_MODEL_PATH}" ]] || { echo "--pretrained-model-path is required for --mode train" >&2; exit 2; }
fi

mkdir -p "${OUTPUT_DIR}"
result="${OUTPUT_DIR}/${MODE}.json"
log="${OUTPUT_DIR}/${MODE}.log"

args=(--checkout "${CHECKOUT}" --mode "${MODE}" --device "${DEVICE}" --attn-mode "${ATTN_MODE}"
      --seed "${SEED}" --output-dir "${OUTPUT_DIR}")
if [[ "${MODE}" == "train" ]]; then
    args+=(--dataset-path "${DATASET_PATH}" --pretrained-model-path "${PRETRAINED_MODEL_PATH}"
           --ngpu "${NGPU}" --num-steps "${NUM_STEPS}" --save-interval "${SAVE_INTERVAL}")
fi
[[ -n "${IFNAME}" ]] && args+=(--ifname "${IFNAME}")

PYTHONPATH="${ROOT}:${PYTHONPATH:-}" python -m mu_space migrate lingbotva --checkout "${CHECKOUT}" >/dev/null
python "${ROOT}/model_examples/LingBotVA/scripts/train_step_musa.py" "${args[@]}" | tee "${log}"

python - "${result}" "${MODE}" <<'PY'
import json
import math
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
mode = sys.argv[2]
errors = []
if payload.get("device") != "musa:0":
    errors.append(f"device is {payload.get('device')!r}, expected musa:0")
if payload.get("optimizer_step") is not True:
    errors.append("optimizer_step is not true")
if payload.get("finite_gradients") is not True:
    errors.append("finite_gradients is not true")
losses = payload.get("losses") or []
if not losses or not all(math.isfinite(float(value)) for value in losses):
    errors.append("losses are missing or not finite")
if mode == "smoke":
    if payload.get("block_mask_gate_all_pass") is not True:
        errors.append(f"compiled BlockMask gate failed: {payload.get('block_mask_gate_errors')}")
    if payload.get("flex_forward_backward_finite") is not True:
        errors.append(f"compiled FlexAttention forward/backward failed: {payload.get('flex_forward_backward_error')}")
else:
    if payload.get("training_completed") is not True:
        errors.append("training did not complete")
    if payload.get("failure_markers"):
        errors.append(f"failure markers in log: {payload['failure_markers']}")
if errors:
    raise SystemExit(f"LingBot-VA {mode} validation failed: " + "; ".join(errors))
print(f"LingBot-VA {mode} validation passed: {sys.argv[1]}")
PY
