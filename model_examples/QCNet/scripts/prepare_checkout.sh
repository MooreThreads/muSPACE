#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY="https://github.com/ZikangZhou/QCNet.git"
readonly COMMIT="55cacb418cbbce3753119c1f157360e66993d0d0"
readonly DESTINATION="${1:-QCNet}"

if [[ ! -d "${DESTINATION}/.git" ]]; then
    git init "${DESTINATION}"
    git -C "${DESTINATION}" remote add origin "${REPOSITORY}"
fi
actual_remote="$(git -C "${DESTINATION}" remote get-url origin)"
[[ "${actual_remote%.git}" == "${REPOSITORY%.git}" ]] || { echo "unexpected origin: ${actual_remote}" >&2; exit 2; }
git -C "${DESTINATION}" fetch --depth 1 --filter=blob:none origin "${COMMIT}"
git -C "${DESTINATION}" checkout --detach "${COMMIT}"
actual_commit="$(git -C "${DESTINATION}" rev-parse HEAD)"
[[ "${actual_commit}" == "${COMMIT}" ]] || { echo "commit verification failed: ${actual_commit}" >&2; exit 3; }
test -f "${DESTINATION}/train_qcnet.py"
echo "QCNet checkout ready: ${DESTINATION} @ ${actual_commit}"
