#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
WORK_DIR=${WORK_DIR:-"${ROOT}/build/openmmlab"}
WHEEL_DIR=${WHEEL_DIR:-"${ROOT}/wheelhouse"}

mkdir -p "${WORK_DIR}" "${WHEEL_DIR}"
python -m pip install setuptools==80.9.0 pytest-runner

cd "${WORK_DIR}"
python -m pip download --no-deps --no-build-isolation --no-binary=:all: mmcv-full==1.4.0
python -m pip download --no-deps --no-build-isolation --no-binary=:all: mmdet3d==0.17.1

rm -rf mmcv-full-1.4.0 mmdet3d-0.17.1
tar -xzf mmcv-full-1.4.0.tar.gz
tar -xzf mmdet3d-0.17.1.tar.gz

patch -p1 -d mmcv-full-1.4.0 \
    < "${ROOT}/model_examples/BEVFormer/patches/deps-mmcv-full-1.4.0.patch"
patch -p1 -d mmdet3d-0.17.1 \
    < "${ROOT}/model_examples/BEVFormer/patches/deps-mmdet3d-0.17.1.patch"

(cd mmcv-full-1.4.0 && MMCV_WITH_OPS=0 python -m build --wheel --no-isolation)
(cd mmdet3d-0.17.1 && MMDET3D_WITH_OPS=0 python -m build --wheel --no-isolation)

cp mmcv-full-1.4.0/dist/*.whl "${WHEEL_DIR}/"
cp mmdet3d-0.17.1/dist/*.whl "${WHEEL_DIR}/"
printf 'Compatibility wheels written to %s\n' "${WHEEL_DIR}"
