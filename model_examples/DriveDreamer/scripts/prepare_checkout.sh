#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY="https://github.com/JeffWang987/DriveDreamer.git"
readonly COMMIT="da1ca92f831bc23d91b59ad418eb47b41cbb1fa9"
readonly DESTINATION="${1:-DriveDreamer}"

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
[[ "${actual_commit}" == "${COMMIT}" ]] || {
    echo "commit verification failed: ${actual_commit}" >&2
    exit 3
}
echo "DriveDreamer checkout ready: ${DESTINATION} @ ${actual_commit}"
