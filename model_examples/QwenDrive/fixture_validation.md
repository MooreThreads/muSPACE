# Qwen-Drive Fixture Validation

The fixture runs one AdamW step on the published Qwen-Drive perception head. It checks that the
head's forward, backward, and optimizer step complete on MUSA with finite head gradients. It is not
a full training run, not a convergence result, and not a throughput benchmark.

## What the check uses

`QwenDrivePerception.infer` is inference-only: it is wrapped in `torch.no_grad` and returns decoded
boxes, occupancy, and map labels. The training step does not call it. It calls `bev_modeling` on the
two feature taps the published head consumes:

- `img_vit_feats`: pre-merge ViT grid, `[N_cam, 32, 56, 1024]` at the released 896×512 image
- `img_llm_feats`: merged language-model image tokens, `[N_cam, 16, 28, 2560]`

Those spatial sizes are the released frustum (`16×16` bins over `896×512`, 118 depth bins). The
fixture does not shrink the BEV grid, the layer counts, or the query count. The config ignores
overrides and keeps the released constants.

The loss is a stand-in, not the paper's detection-matching loss. Classification, box, occupancy, and
map outputs contribute a mean-square penalty on their logits. In real mode, occupancy or map is
replaced by cross-entropy only when `gt.npz` contains a label grid whose shape matches that output.
Any other key, or a grid that does not match, keeps the logit penalty. Box outputs are never matched
to ground-truth boxes.

## Prerequisites

Install muSPACE and prepare the pinned checkout from `docs/install.md`:

```bash
model_examples/QwenDrive/scripts/prepare_checkout.sh ./QwenDrive
mu-space migrate qwendrive --checkout ./QwenDrive
```

`mu-space migrate qwendrive` writes `.mu_space_qwendrive.json` and does not modify the checkout. The
script applies `qwendrive_patcher` itself. Run it where `mu-space env` reports a MUSA device.
`--device` must be a MUSA device. The published attention path is bfloat16, and so is this step.

Perception-side packages that are not part of the MUSA stack are listed in `requirements.lock`. Do
not install the upstream CUDA pins (torch 2.8, flash-attn, flash-linear-attention, causal-conv1d);
they replace the MUSA build. `--attn-implementation` defaults to `sdpa` for the same reason.

## One-command check

Synthetic features, no VLM weights:

```bash
python model_examples/QwenDrive/scripts/train_step_musa.py \
  --checkout ./QwenDrive --mode fixture --device musa:0
```

Strict mode requires the native MUSA kernels and fails if dispatch falls back:

```bash
python model_examples/QwenDrive/scripts/train_step_musa.py \
  --checkout ./QwenDrive --mode fixture --device musa:0 --strict
```

Repeat the same head step on consecutive devices. Each card runs the step in turn on one shared head
copy. The script selects that card as the current MUSA device before moving the head onto it. Card 0
is not a special case in fixture mode. This does not start a data-parallel job:

```bash
python model_examples/QwenDrive/scripts/train_step_musa.py \
  --checkout ./QwenDrive --mode fixture --device musa:0 --device-count 4 --strict
```

Frozen VLM, one packed demo frame. `<VLM_ROOT>` is a Qwen-Drive-1.0-4B directory whose `perception/`
subdirectory holds the head weights. `<DEMO_FRAMES>` contains one directory per frame
(`frame.json`, `images/`, `calib.npz`, `gt.npz`). The published loader opens the last two, so a
frame missing either is rejected before the model is imported. A label grid inside `gt.npz` replaces
the logit penalty only when its shape matches. The VLM forward runs once on the first
device; each device then takes its own head step on the detached taps:

```bash
python model_examples/QwenDrive/scripts/train_step_musa.py \
  --checkout ./QwenDrive --mode real --device musa:0 \
  --vlm <VLM_ROOT> --perception <VLM_ROOT>/perception --frames <DEMO_FRAMES>
```

The script prints one JSON object on stdout. It does not write a result file.

## Acceptance criteria

The check passes only when that JSON contains:

- `status` equal to `qwendrive_perception_train_step_ok`
- `mode` of `fixture` or `real`
- one `steps` entry per requested device
- a finite numeric `loss` and a positive finite `head_grad_norm` in every step
- `head_finite_gradients: true` and `optimizer_step: true` in every step
- `backbone_grad_is_none: true`

In fixture mode `backbone_present` is false: no VLM was loaded, so there is no backbone gradient to
produce. In real mode `backbone_present` is true and every VLM parameter gradient is `None`. A
produced head gradient that is non-finite, or a head that produces no gradient, fails the process
before this JSON is printed. A head parameter the stand-in loss does not reach may keep a `None`
gradient.

`elapsed_seconds` and `peak_memory_bytes` are diagnostic. `peak_memory_bytes` on the first device in
real mode includes the frozen VLM, because that forward stays resident for the gradient check. Do
not quote either field as throughput or as a memory target.

`MUSPACE_DISPATCH_LOG=1` prints each operator's backend choice to stderr. In strict mode the native
MUSA backend is required for both multi-scale deformable attention and depth-aware voxel pooling.

## Troubleshooting

Argument errors (missing checkout, unknown `--mode`, `--device-count` below 1, a non-MUSA
`--device`, a real-mode path that is not a directory, or a packed frame missing `calib.npz` or
`gt.npz`) exit 2 before the model is imported. The checkout must contain
`src/qwen_drive_perception`.

If the perception package fails while importing its CUDA extension loader, confirm the script is the
one applying `qwendrive_patcher` before the head is built. The published kernels are not the ones
this step runs. A missing or partial package raises `PatchError` and names the symbol that was not
found.

If fixture mode passes and real mode fails inside the VLM forward, the head step itself is not the
failure. The VLM is frozen and is not part of the backward. Fix the VLM forward, or keep using
fixture mode, which never loads it.

If default mode passes and `--strict` fails, an operator fell back to the reference path. Rerun the
multi-scale deformable attention and `voxel_pool_depth` operator tests before treating the head step
as the defect. If the loss or a head gradient is non-finite, rerun with the same `--seed`.
