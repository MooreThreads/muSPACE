#!/usr/bin/env bash
# Build the OpenMMLab compatibility wheels this recipe needs on Python 3.10.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WHEEL_DIR="${WHEEL_DIR:-${ROOT}/wheelhouse}"
PATCHES="${ROOT}/model_examples/VAD/patches"

mkdir -p "${WHEEL_DIR}"
python -m pip download --no-deps --no-build-isolation --no-binary=:all: mmcv==1.7.0 \
    --dest "${WHEEL_DIR}"

rm -rf mmcv-1.7.0
tar -xzf "${WHEEL_DIR}/mmcv-1.7.0.tar.gz"
patch -p1 -d mmcv-1.7.0 < "${PATCHES}/deps-mmcv-1.7.0-lazy-ext.patch"
(cd mmcv-1.7.0 && MMCV_WITH_OPS=0 python -m build --wheel --no-isolation)
cp mmcv-1.7.0/dist/*.whl "${WHEEL_DIR}/"
rm -rf mmcv-1.7.0

cat <<'USAGE'
Built wheelhouse/mmcv-1.7.0-*.whl with the lazy extension loader.

Install it with:
  pip install --no-deps wheelhouse/mmcv-1.7.0-*.whl

The remaining OpenMMLab packages follow the same compatibility workflow as the
UniAD recipe: mmdet 2.28.0, mmseg 0.30.0 and the mmdet3d 1.0.0rc4 checkout that
is installed from the pinned source tree, then
  pip install -r model_examples/VAD/requirements.lock
USAGE
