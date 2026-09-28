#!/usr/bin/env bash
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
set -euo pipefail

readonly REPOSITORY="https://github.com/Robbyant/lingbot-va.git"
readonly COMMIT="58c2ae5bac46bd8114065bea9d7d256eb67c16c3"
readonly DESTINATION="${1:-LingBotVA}"

if [[ ! -d "${DESTINATION}/.git" ]]; then
    git init "${DESTINATION}"
    git -C "${DESTINATION}" remote add origin "${REPOSITORY}"
fi

actual_remote="$(git -C "${DESTINATION}" remote get-url origin)"
if [[ "${actual_remote%.git}" != "${REPOSITORY%.git}" ]]; then
    echo "unexpected origin: ${actual_remote}" >&2
    exit 2
fi

git -C "${DESTINATION}" fetch --depth 1 --filter=blob:none origin "${COMMIT}"
git -C "${DESTINATION}" checkout --detach "${COMMIT}"

actual_commit="$(git -C "${DESTINATION}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${COMMIT}" ]]; then
    echo "commit verification failed: ${actual_commit}" >&2
    exit 3
fi

test -f "${DESTINATION}/wan_va/train.py"
echo "LingBot-VA checkout ready: ${DESTINATION} @ ${actual_commit}"
