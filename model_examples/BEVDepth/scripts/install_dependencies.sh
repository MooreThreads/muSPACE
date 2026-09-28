#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON="${PYTHON:-python}"

# Lightning 1.6.5 and mmdet3d 1.0.0rc4 carry legacy metadata that modern pip
# cannot solve on Python 3.10. Install the platform-independent pins without
# dependency resolution, then install the separately pinned runtime closure.
"${PYTHON}" -m pip install 'pip==23.3.1' 'setuptools==59.5.0'
"${PYTHON}" -m pip install --no-build-isolation --no-deps \
    -r "${ROOT}/model_examples/BEVDepth/requirements.lock"
"${PYTHON}" -m pip install --user --no-deps \
    -r "${ROOT}/model_examples/BEVDepth/requirements-runtime.lock"
