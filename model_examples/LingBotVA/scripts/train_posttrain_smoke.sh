#!/usr/bin/env bash
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#
# MUSA post-training launcher for LingBot-VA.  The environment mirrors the
# validated single-node configuration: one optimizer step per
# `GRADIENT_ACCUMULATION_STEPS` micro-batches, `batch size` samples per rank.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CHECKOUT="${CHECKOUT:-LingBotVA}"
DATASET_PATH="${DATASET_PATH:-}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-}"
NGPU="${NGPU:-8}"
MBS="${MBS:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-10}"
NUM_STEPS="${NUM_STEPS:-1}"
ATTN_MODE="${ATTN_MODE:-flex}"
SEED="${SEED:-20260906}"
SAVE_ROOT="${SAVE_ROOT:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/.tmp/lingbotva_train}"
IFNAME="${IFNAME:-}"
DATASET_INIT_WORKER="${DATASET_INIT_WORKER:-1}"
LOAD_WORKER="${LOAD_WORKER:-16}"

if [[ -z "${DATASET_PATH}" || -z "${PRETRAINED_MODEL_PATH}" ]]; then
    echo "DATASET_PATH and PRETRAINED_MODEL_PATH are required" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
args=(--checkout "${CHECKOUT}" --mode train --device musa:0 --attn-mode "${ATTN_MODE}"
      --seed "${SEED}" --output-dir "${OUTPUT_DIR}"
      --dataset-path "${DATASET_PATH}" --pretrained-model-path "${PRETRAINED_MODEL_PATH}"
      --ngpu "${NGPU}" --num-steps "${NUM_STEPS}" --batch-size "${MBS}"
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
      --dataset-init-worker "${DATASET_INIT_WORKER}" --load-worker "${LOAD_WORKER}"
      --save-interval "${SAVE_INTERVAL}")
[[ -n "${SAVE_ROOT}" ]] && args+=(--save-root "${SAVE_ROOT}")
[[ -n "${IFNAME}" ]] && args+=(--ifname "${IFNAME}")

PYTHONPATH="${ROOT}:${PYTHONPATH:-}" python -m mu_space migrate lingbotva --checkout "${CHECKOUT}" >/dev/null
python "${ROOT}/model_examples/LingBotVA/scripts/train_step_musa.py" "${args[@]}"
