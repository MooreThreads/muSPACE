# QCNet for muSPACE

## Contents

- [Overview](#overview)
- [Preparing the environment](#preparing-the-environment)
- [Preparing the model and data](#preparing-the-model-and-data)
- [S5000 training smoke](#s5000-training-smoke)
- [A100 precision alignment](#a100-precision-alignment)
- [Results and limitations](#results-and-limitations)

## Overview

This recipe integrates the official QCNet repository at commit
`55cacb418cbbce3753119c1f157360e66993d0d0`. muSPACE does not maintain a model fork.
The migration patch replaces only QCNet's `torch_cluster.radius` imports with a device-aware
compatibility layer: MUSA tensors require the native muSPACE implementation, while CUDA and CPU
tensors continue to use the upstream `torch_cluster` implementation.

The validation boundary is one or more optimizer steps on a small, real Argoverse 2 motion
forecasting subset. It is a launch and numerical-alignment check, not full 64-epoch convergence or
a leaderboard claim.

| Component | Requirement |
| --- | --- |
| QCNet | `55cacb418cbbce3753119c1f157360e66993d0d0` |
| Dataset | Argoverse 2 Motion Forecasting |
| MUSA device | MTT S5000 |
| Dtypes | FP32 and FP16 mixed precision |
| muSPACE operators | `radius`, `radius_graph` |

## Preparing the environment

Install a matching PyTorch and torch_musa stack, then install the model dependencies without
replacing the platform PyTorch package:

```bash
pip install -r model_examples/QCNet/requirements.lock
```

The lock intentionally does not declare PyTorch, so an already installed matching platform build
is preserved. PyG extension wheels are platform-specific. A MUSA run does not require the CUDA
`torch-cluster` extension for radius queries. The A100 baseline must install a torch-cluster wheel
matching its CUDA/PyTorch build because it intentionally uses the upstream operator.

Build and install the current muSPACE wheel before validation so the native library contains the
radius entrypoints.

## Preparing the model and data

Prepare and migrate the pinned checkout:

```bash
model_examples/QCNet/scripts/prepare_checkout.sh ./QCNet
mu-space migrate qcnet --checkout ./QCNet
```

Download Argoverse 2 following its public instructions. If the complete dataset is laid out as
`<AV2_ROOT>/<split>/raw/<scenario-id>`, create a small read-only subset:

```bash
python model_examples/QCNet/scripts/prepare_subset.py \
  --source-root <AV2_ROOT> --output-root <AV2_SUBSET_ROOT> --split train --count 2
```

The helper creates symbolic links and never changes the source dataset. QCNet performs its normal
preprocessing when the training launcher first opens the subset.

## S5000 training smoke

Run FP32 and FP16 independently:

```bash
model_examples/QCNet/scripts/validate_fixture.sh \
  --checkout ./QCNet --data-root <AV2_SUBSET_ROOT> --dtype float32 \
  --output-dir .tmp/qcnet-fp32

model_examples/QCNet/scripts/validate_fixture.sh \
  --checkout ./QCNet --data-root <AV2_SUBSET_ROOT> --dtype float16 \
  --output-dir .tmp/qcnet-fp16
```

The launcher uses the upstream QCNet model and training step directly on one card, batch size one,
and one optimizer step by default. Direct single-card execution keeps MUSA device placement
explicit because the upstream Lightning auto-accelerator does not identify every torch_musa build.
Each output directory contains the serialized real batch, identical
initial state, trained state, and a JSON result. See [fixture_validation.md](fixture_validation.md)
for the acceptance contract.

## A100 precision alignment

Generate the A100 artifact from the exact batch and initial state captured on S5000. The CUDA
environment must use the same QCNet commit and compatible dependency versions:

```bash
python model_examples/QCNet/scripts/precision_compare.py \
  --checkout ./QCNet --batch <MUSA_OUTPUT>/batch.pt \
  --state <MUSA_OUTPUT>/initial_state.pt --device cuda:0 --dtype float32 \
  --output <A100_OUTPUT>/qcnet-fp32.pt
```

Generate the S5000 artifact and compare it with the A100 artifact:

```bash
python model_examples/QCNet/scripts/precision_compare.py \
  --checkout ./QCNet --batch <MUSA_OUTPUT>/batch.pt \
  --state <MUSA_OUTPUT>/initial_state.pt --device musa:0 --dtype float32 \
  --reference <A100_OUTPUT>/qcnet-fp32.pt --output <MUSA_OUTPUT>/qcnet-fp32.pt
```

Repeat with `--dtype float16`. Default tolerances are `atol=3e-4, rtol=1e-3` for FP32 and
`atol=5e-3, rtol=5e-2` for FP16. The JSON report records per-output maximum absolute and relative
errors, mean absolute error, mismatch count, and worst flat index.

## Results and limitations

No S5000 training or A100 alignment result is published until both environments execute the
commands above successfully. Training launch, numerical alignment, and performance are separate
claims; this recipe does not contain a performance benchmark.

- Full-dataset preprocessing and 64-epoch convergence are outside this smoke-test boundary.
- FP16 uses torch autocast with explicit loss scaling and depends on the installed torch_musa integration.
- MUSA validation is strict: native radius failure is not silently replaced by torch-cluster.
- CUDA comparison uses the original torch-cluster operator, not the muSPACE reference function.
