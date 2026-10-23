#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY="https://github.com/QwenLM/Qwen-Drive-1.0.git"
readonly COMMIT="28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0"
readonly DESTINATION="${1:-QwenDrive}"

if [[ ! -d "${DESTINATION}/.git" ]]; then
    git init "${DESTINATION}"
    git -C "${DESTINATION}" remote add origin "${REPOSITORY}"
fi

actual_remote="$(git -C "${DESTINATION}" remote get-url origin)"
if [[ "${actual_remote%.git}" != "${REPOSITORY%.git}" ]]; then
    echo "unexpected origin: ${actual_remote}" >&2
    exit 2
fi

git -C "${DESTINATION}" fetch --depth 1 origin "${COMMIT}"
git -C "${DESTINATION}" checkout --detach "${COMMIT}"

actual_commit="$(git -C "${DESTINATION}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${COMMIT}" ]]; then
    echo "commit verification failed: ${actual_commit}" >&2
    exit 3
fi

echo "QwenDrive checkout ready: ${DESTINATION} @ ${actual_commit}"
