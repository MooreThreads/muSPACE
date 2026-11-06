#!/usr/bin/env bash
set -euo pipefail

readonly REPOSITORY="https://github.com/HorizonRobotics/Sparse4D.git"
readonly COMMIT="4ec96edf0e7d9e80fbd75ef22e1d51a937d3a750"
readonly TAG="v3.0"
readonly DESTINATION="${1:-Sparse4D}"

if [[ ! -d "${DESTINATION}/.git" ]]; then
    git init "${DESTINATION}"
    git -C "${DESTINATION}" remote add origin "${REPOSITORY}"
fi

actual_remote="$(git -C "${DESTINATION}" remote get-url origin)"
if [[ "${actual_remote%.git}" != "${REPOSITORY%.git}" ]]; then
    echo "unexpected origin: ${actual_remote}" >&2
    exit 2
fi

if ! git -C "${DESTINATION}" fetch --depth 1 origin "${COMMIT}"; then
    git -C "${DESTINATION}" fetch --depth 1 origin "tag" "${TAG}"
fi
git -C "${DESTINATION}" checkout --detach "${COMMIT}"

actual_commit="$(git -C "${DESTINATION}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${COMMIT}" ]]; then
    echo "commit verification failed: ${actual_commit}" >&2
    exit 3
fi

for required in \
    "projects/mmdet3d_plugin/ops/__init__.py" \
    "tools/nuscenes_converter.py" \
    "tools/anchor_generator.py"; do
    if [[ ! -f "${DESTINATION}/${required}" ]]; then
        echo "checkout is missing ${required}" >&2
        exit 3
    fi
done

echo "Sparse4D checkout ready: ${DESTINATION} @ ${actual_commit}"
