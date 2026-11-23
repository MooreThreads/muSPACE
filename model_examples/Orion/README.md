# Orion for muSPACE

## Contents

- [Overview](#overview)
  - [Model description](#model-description)
  - [Code implementation](#code-implementation)
- [Preparing the environment](#preparing-the-environment)
  - [Install dependencies](#install-dependencies)
  - [Install the MUSA stack](#install-the-musa-stack)
  - [Prepare the checkout](#prepare-the-checkout)
  - [Prepare pretrained weights](#prepare-pretrained-weights)
  - [Prepare the dataset](#prepare-the-dataset)
- [Quick start](#quick-start)
  - [Single-frame inference](#single-frame-inference)
- [Coverage and limitations](#coverage-and-limitations)
- [Changelog](#changelog)
- [FAQ](#faq)

# Overview

## Model description

Orion is a vision-language end-to-end autonomous-driving model. A QT-Former aggregates a
queued multi-camera history, a language model reasons over the collected tokens, and a
detection/map/planning head set emits 3-D boxes (with motion), map polylines, and the ego
trajectory. That inference graph exercises rotated BEV IoU and NMS, deterministic point-in-box
queries, and sigmoid focal loss, all of which muSPACE provides natively on MUSA.

## Code implementation

- Reference implementation:

  ```shell
  url=https://github.com/xiaomi-mlab/Orion.git
  commit_id=657f3cf09b745d3972affb84d68f7e376140b221
  ```

- muSPACE recipe:

  ```shell
  code_path=model_examples/Orion
  ```

muSPACE maintains no model fork. The recipe pins the upstream commit, applies the
compatibility patches under `patches/`, installs the SDPA flash-attention shim, and rebinds the
vendored mmcv operators to muSPACE at runtime.

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Upstream repository, pinned commit, patch marker, and patch digests |
| `requirements.lock` | Pinned Python stack for the verified environment |
| `patches/` | Lazy-import patches for the vendored mmcv extensions and the MUSA device-call fixes |
| `scripts/prepare_checkout.sh` | Fetches and verifies the pinned upstream commit |
| `scripts/infer_frame_musa.py` | Single-frame inference fixture |
| `fixture_validation.md` | Recorded validation of the fixture |

# Preparing the environment

## Install dependencies

```shell
cd model_examples/Orion
python -m pip install -r requirements.lock
```

Keep `numpy<2`: torch 2.11 links against the NumPy 1.x C API, and installing a package such as
`diffusers` unpinned upgrades NumPy to 2.x and breaks the tensor bridge.

## Install the MUSA stack

Build and install the muSPACE wheel for the active MUSA toolkit:

```shell
export MUSA_ARCH=31
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

## Prepare the checkout

```shell
bash model_examples/Orion/scripts/prepare_checkout.sh ./Orion
cd ./Orion && git apply ../../model_examples/Orion/patches/deps-mmcv-ext-loader-lazy.patch \
  ../../model_examples/Orion/patches/orion-01-lazy-iou3d-cuda.patch \
  ../../model_examples/Orion/patches/orion-02-lazy-roiaware-ext.patch \
  ../../model_examples/Orion/patches/orion-03-musa-device-calls.patch
```

The first three patches make the vendored compiled extensions lazy; the fourth replaces the
hardcoded `.cuda()` calls and the `is_cuda` assertion. Neither changes operator semantics; see
[Coverage and limitations](#coverage-and-limitations).

## Prepare pretrained weights

```shell
HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download poleyzdk/Orion Orion.pth --local-dir ckpts
huggingface-cli download poleyzdk/Orion --include 'pretrain_qformer/*' --local-dir ckpts
```

Set `llm_path` in the config to the `pretrain_qformer` directory.

## Prepare the dataset

The reference dataset is [Bench2Drive](https://huggingface.co/datasets/rethinklab/Bench2Drive).
Open-loop evaluation needs the per-clip archives plus `b2d_infos_val.pkl` and
`b2d_map_infos.pkl` under `data/`.

# Quick start

## Single-frame inference

```shell
export PYTHONPATH=$PWD:$PWD/Orion
python model_examples/Orion/scripts/infer_frame_musa.py \
    --checkout ./Orion \
    --checkpoint ckpts/Orion.pth \
    --llm-path ckpts/pretrain_qformer
```

The fixture builds the published `orion_stage3_infer` configuration, feeds one synthetic frame
shaped like the pipeline output, and prints a JSON summary. It succeeds when every prediction is
finite and at least one replaced operator reported the `musa` backend.

`--llm-path` is required: upstream reads `self.lm_head` whenever a tokenizer is configured, so the
model cannot be built without it.

To run one real Bench2Drive frame instead of the synthetic one:

```shell
python model_examples/Orion/scripts/infer_frame_musa.py \
    --checkout ./Orion \
    --checkpoint ckpts/Orion.pth \
    --llm-path ckpts/pretrain_qformer \
    --input bench2drive --clip-dir data/clips/<clip> --frame-id 00000
```

For a steady-state latency figure, add `--benchmark 5`. Add `--lm-fp16` to run only the LM head in
float16, which is the recipe's only supported reduced-precision path: it cuts latency and peak
memory substantially, while the published `orion_stage3_fp16.py` is unusable because
`img_backbone.half()` collides with the attention layer's `out_fp32=True`. Measured numbers and the
full list of upstream defects are in `fixture_validation.md`.

# Coverage and limitations

The recipe covers single-frame inference only:

- No training. The three-stage training pipeline and the multi-card runs are out of scope; the
  published LoRA adapters are loaded for inference.
- No data pipeline. The fixture reads either synthetic tensors or a single clip frame; dataset-side
  evaluation is not automated here.
- No multi-card execution and no Bench2Drive closed-loop metrics.
- No numerical alignment of the predictions against the upstream CUDA results.
- Reduced precision only via `--lm-fp16`, which casts the LM head alone. The published
  `orion_stage3_fp16.py` does not run; see `fixture_validation.md`.

Operator coverage: rotated BEV IoU and NMS (`boxes_iou_bev`, `nms_bev`, `nms_bev_normal`), the
point-in-box queries, and sigmoid focal loss run on muSPACE kernels. The remaining compiled mmcv
extension entry points (`deform_conv`, `masked_conv`, `roi_align`, `nms`, `voxelize`, the
deformable-conv family) stay behind the lazy stand-in and raise a descriptive error if a model
path reaches them.

# Changelog

- 2026-11-16: real-data and FP16 validation, steady-state latency and peak-memory matrix on
  S5000 and A100, plus the `--lm-fp16` path and the upstream-defect notes.
- 2026-11-11: single-frame inference recipe with the vendored mmcv rebinding and the SDPA
  flash-attention shim.

# FAQ

**Why is the flash-attention shim installed before importing mmcv?**
Orion's `mmcv/models/utils/attention.py` imports `flash_attn_varlen_kvpacked_func` at module
import time. The shim exposes that name (and the padding helpers) backed by
`torch.nn.functional.scaled_dot_product_attention`, which runs natively on MUSA.

**Why patch the vendored mmcv instead of shipping a fork?**
The three patches only defer extension imports; every operator binding is installed at runtime by
`mu_space.patcher.orion_patcher`, so the checkout stays byte-identical to upstream apart from the
lazy imports.
