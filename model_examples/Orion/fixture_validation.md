# Orion fixture validation

This file records the validation of the Orion single-frame fixture shipped with this recipe.
Every table below reports measured results; work that has not run is listed under
[Outstanding](#outstanding).

## Environment

### MTT S5000

| Item | Value |
| --- | --- |
| Torch stack | torch 2.11.0 + torch_musa 2.11.0 |
| Python | 3.10 |
| NumPy | 1.23.5 (torch links against the NumPy 1.x C API) |
| Transformers | 4.31.0 (`peft` 0.12.0, `huggingface-hub` 0.23.5) |
| MUSA arch | `MUSA_ARCH=31` |
| Attention | `mu_space.patcher.orion_flash_attn` SDPA shim |

### NVIDIA A100 (comparison baseline)

| Item | Value |
| --- | --- |
| Container | dedicated task container |
| GPU | A100 80GB PCIe |
| Torch stack | torch 2.11.0+cu128, CUDA 12.8 |
| Attention | upstream `flash-attn` 2.8.3 |

The musa side needs no CUDA, and the A100 side needs no MUSA: the two runs share the checkpoint,
the prompt tokens, and the fixture geometry, and differ only in device and operator backend.

## Vendored mmcv import

The pinned checkout cannot import before the lazy-extension patches are applied:

| Check | Result |
| --- | --- |
| `import mmcv` without patches | fails on `mmcv._ext` |
| `+ deps-mmcv-ext-loader-lazy.patch` | fails on `iou3d_cuda` |
| `+ orion-01-lazy-iou3d-cuda.patch` | fails on `roiaware_pool3d_ext` |
| `+ orion-02-lazy-roiaware-ext.patch` and the SDPA shim | imports |

Extension entry points without a muSPACE replacement stay behind `MissingExtension` and raise a
`RuntimeError` naming the kernel when called.

The same holds on A100: the upstream tree has no compiled `mmcv` extensions there either, so that
host runs the recipe too, with the patcher bound to the pure-PyTorch `reference` backend.

## Flash-attention shim

`mu_space.patcher.orion_flash_attn` provides `flash_attn_varlen_kvpacked_func` and the padding
helpers on `torch.nn.functional.scaled_dot_product_attention`.

| Check | Result |
| --- | --- |
| Single packed sequence vs float64 reference | within `1e-5` |
| Ragged batch (query lengths 3/1, key lengths 3/5) | within `1e-5` |
| Padded key tail does not leak into the result | within `1e-5` |

## Device-agnostic calls

`patches/orion-03-musa-device-calls.patch` removes the hardcoded accelerator assumptions:

| Site | Change |
| --- | --- |
| `mmcv/models/utils/attention.py` | assert accepts `musa`/`privateuseone` beside `cuda` |
| `mmcv/models/utils/grid_mask.py` | mask built on the input device instead of `.cuda()` |
| `mmcv/models/dense_heads/orion_head.py` | NMS inputs and known mask follow their source tensors |
| `mmcv/models/detectors/orion.py` | dummy waypoint ids follow the LM head device |

## Inference configuration

The fixture keeps the upstream inference defaults from `orion_stage3_infer.py`:

| Setting | Value | Why |
| --- | --- | --- |
| `use_lora` | `True` | The published checkpoint stores LoRA adapters |
| `use_gen_token` | `True` | Adds `<waypoint_ego>`; the vocabulary grows to 32001, matching the checkpoint |
| `fp32_infer` | `True` | The fixture has no `gt_attr_labels`, so the metric block is skipped |

The prompt comes from the upstream `LoadAnnoatationCriticalVQATest` transform with
`use_gen_token=True`, which ends the conversation in `<waypoint_ego>` (id 32000). That token marks
the sample as a final generation round; a hand-built placeholder sequence does not carry it and
leaves `ego_fut_preds` unassigned in `simple_test_pts`.

`DEBUG_SHOW_DIR_PRED` is set so the class-agnostic NMS pass in `get_motion_bboxes` executes.
Without it the decoded boxes skip NMS and no replaced operator runs.

## Single-frame forward

| Input | Checkpoint | Finite | Dispatch | Operators |
| --- | --- | --- | --- | --- |
| Synthetic frame | `Orion.pth` | yes (8 outputs) | 1 MUSA | `nms_bev` |
| Bench2Drive `AccidentTwoWays_Town12_Route1102_Weather10` frame `00000` | `Orion.pth` | yes (8 outputs) | 1 MUSA | `nms_bev` |

`Orion.pth` loads with `missing=0` in both runs. The Bench2Drive run reads the six
`camera/rgb_*` images of one frame from a single clip; the temporal queue is zero-filled and the
can-bus values come from that frame's `anno` entry.

## Performance

Protocol: build once, warm up one forward, then time `--benchmark` repetitions that each reset
`test_flag` and the detector/map memory banks first, so a sample is the cost of one independent
frame rather than a growing history queue. Both hosts run the same checkpoint, the same 71-token
prompt, and the same seed-0 geometry.

### Steady-state latency and peak memory

| Input | Precision | S5000 | A100 | S5000 advantage |
| --- | --- | --- | --- | --- |
| Synthetic | FP32 | **1.229 s** / 28.92 GB | 1.363 s / 28.93 GB | 9.8% |
| Bench2Drive | FP32 | **1.198 s** / 28.92 GB | 1.398 s / 28.93 GB | 14.3% |
| Synthetic | FP16 LM head | **0.751 s** / 16.05 GB | 0.920 s / 16.03 GB | 18.4% |
| Bench2Drive | FP16 LM head | **0.761 s** / 16.05 GB | 0.952 s / 16.03 GB | 20.1% |

Per-run samples, S5000 (five repetitions):

| Input | Precision | Samples | Spread |
| --- | --- | --- | --- |
| Synthetic | FP32 | 1.243 / 1.216 / 1.208 / 1.239 / 1.237 | 2.9% |
| Bench2Drive | FP32 | 1.214 / 1.216 / 1.213 / 1.176 / 1.171 | 3.8% |
| Synthetic | FP16 | 0.763 / 0.755 / 0.782 / 0.740 / 0.716 | 9.2% |
| Bench2Drive | FP16 | 0.762 / 0.764 / 0.763 / 0.757 / 0.760 | 0.9% |

Two observations:

- The synthetic and real inputs land within run-to-run noise of each other on S5000, so the
  synthetic fixture is representative for latency. It is not representative for correctness;
  see [Upstream defects](#upstream-defects).
- Peak memory matches across hosts to within 0.03 GB, which confirms both runs load the same
  weights and traverse the same graph.

### FP16

Casting the LM head to float16 cuts latency by 1.64x on S5000 and 1.48x on A100, and peak memory
by 44.5% on both. The resulting 16.05 GB agrees with the upstream claim of "more than 17GB" for
FP16 inference.

The cast has to happen after `load_state_dict`, otherwise the float32 checkpoint overwrites it and
the run silently shows no gain.

### Fairness caveats

Two deviations pull in opposite directions and cannot be quantified exactly:

| Item | S5000 | A100 | Disadvantaged |
| --- | --- | --- | --- |
| BEV NMS | native MUSA kernel | muSPACE `reference` (PyTorch, no compiled extension) | A100 |
| Attention | SDPA shim | real `flash-attn` 2.8.3 | S5000 |

NMS is roughly 3% of a frame (0.036 s of 1.2 s). The attention difference is the larger of the
two, so the quoted S5000 advantage is a conservative figure: a baseline with compiled `mmcv`
extensions would narrow it.

## Upstream defects

Four defects in the pinned commit surfaced while validating. The recipe works around all of them
without patching model semantics.

| # | Site | Symptom | Workaround |
| --- | --- | --- | --- |
| 1 | `orion.py:198` | `self.lm_head` is read whenever a tokenizer is configured, so `lm_head=None` raises | always pass `--llm-path` |
| 2 | `orion.py:196` | with `use_gen_token` off, `add_special_token` never runs, the LM head stays at 32000 rows, and the checkpoint reports three size mismatches | keep `use_gen_token=True` |
| 3 | `extract_img_feat` | `img.squeeze_()` mutates the caller's tensor in place; `simple_test` then re-adds a batch dim and the camera/position counts drift (1600 vs 9600) | feed a 6-D `[B, T, N, C, H, W]` stack |
| 4 | `orion_stage3_fp16.py` | `img_backbone.half()` conflicts with `@auto_fp16(..., out_fp32=True)` on the attention layer, which returns float32 into float16 weights | cast only the LM head (`--lm-fp16`) |

Defect 4 makes the published FP16 configuration unusable as shipped, so the recipe does not offer
`orion_stage3_fp16.py`. Defect 3 is only reachable on the real-data path; the synthetic fixture
happened to dodge it, which is why the real Bench2Drive frame is part of the validation.

## Test suite

```
python -m pytest tests/test_orion_patcher.py -q
7 passed
```

`tests/test_orion_patcher.py` covers the shim numerics, the patcher rebinding and restore, the
backend validation, and the missing-module error.

## Outstanding

- No Bench2Drive evaluation split and no closed-loop metrics. That needs the per-clip archives
  plus `b2d_infos_val.pkl` and `b2d_map_infos.pkl`.
- No numerical alignment of the predictions against the upstream CUDA reference.
- Only the `orion_stage3_infer` configuration is covered. `orion_stage3_cot` and the FP16 config
  are out of scope, the latter because of defect 4.
- Training, multi-card execution, and the data pipeline remain out of scope by design.
