#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY="https://github.com/openvla/openvla.git"
readonly COMMIT="c8f03f48af692657d3060c19588038c7220e9af9"
readonly DESTINATION="${1:-OpenVLA}"

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

test -f "${DESTINATION}/vla-scripts/finetune.py"
echo "OpenVLA checkout ready: ${DESTINATION} @ ${actual_commit}"
