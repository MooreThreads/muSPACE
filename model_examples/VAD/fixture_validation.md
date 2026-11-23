# VAD Fixture Validation

The fixture validation runs one small VAD training iteration and one single-frame inference
forward with the muSPACE multi-scale deformable attention patch. Both are launch and correctness
checks for the muSPACE VAD integration path, not full training, accuracy, or performance results.

## Prerequisites

Install muSPACE, prepare the pinned VAD checkout, apply the migration recipe, and point
`--data-root` at a nuScenes root that contains the VAD temporal infos
(`vad_nuscenes_infos_temporal_train.pkl`, `vad_nuscenes_infos_temporal_val.pkl`). The launch checks
select one scene from the infos at runtime, so only that scene's sensor files are needed while the
version metadata (`<version>/scene.json`) must be complete. CAN bus and map expansion files are
required in either case.

```bash
mu-space migrate vad --checkout ./VAD
```

Run inside the validated MUSA environment reported by `mu-space env --strict`. Both fixtures apply
the muSPACE patch in strict mode before building the model, so a run that falls back to the PyTorch
reference path fails instead of silently passing. Each fixture also asserts that VAD's
`MultiScaleDeformableAttnFunction_fp32`/`_fp16` entries were rebound to muSPACE.

Both fixtures call `model.init_weights()` after building the model. Upstream initialises the
transformer with `xavier_uniform`; without that call the BEV encoder activations grow by roughly
an order of magnitude per layer and overflow float32, so the fixtures fail on a fraction of the
random initialisations.

## Training Fixture

```bash
bash model_examples/VAD/scripts/train_1p_stage1_smoke.sh --checkout ./VAD --data-root /path/to/nuscenes
```

The launcher defaults to `projects/configs/VAD/VAD_base_stage_1.py`, the first scene present in the
infos, one iteration, device `musa:0`, and `<checkout>/data/nuscenes` as the data root.
`--pretrained-backbone` additionally initialises the ResNet backbone from the ImageNet weights;
it needs network access on first use and is off by default. Each default
is overridable by flag (`--config`, `--scene-name`, `--max-iters`, `--device`, `--data-root`,
`--work-dir`) or the matching environment variable. Scratch metadata is written below the system
temporary location by default; pass `--work-dir` explicitly for a controlled location (for
repository-local validation, use `<repository>/.tmp`). The optimizer is built from the config
through OpenMMLab's `build_optimizer`, including its configured parameter rules.

A successful run prints one line:

```
vad_train_step_ok optimizer=AdamW param_groups=<N> loss=<finite> grad_norm=<finite> param_delta=<positive> logs={...}
```

## Inference Fixture

```bash
python model_examples/VAD/scripts/infer_frame_musa.py --checkout ./VAD --data-root /path/to/nuscenes \
    --device musa:0
```

The fixture builds the test dataset, takes one prepared frame, runs the image backbone and the VAD
head under `torch.no_grad()`, and checks that the prediction tensors are finite with the expected
shapes. The output names the validated groups:

```
vad_infer_ok device=musa:0 scene=<name> report={...}
```

`report` covers detection (`all_cls_scores`, `all_bbox_preds`), map (`map_all_cls_scores`,
`map_all_pts_preds`) and agent trajectory (`all_traj_preds`) tensors. The ego-conditioned planning
branch (`ego_fut_preds`) is fed by the stage-2/e2e planner inputs and is outside this fixture's
scope.

## Acceptance Criteria

The validation passes only when:

- the training fixture prints `vad_train_step_ok` with a finite loss, a finite gradient norm, and a
  positive parameter update,
- the inference fixture prints `vad_infer_ok` with finite detection, map, and trajectory tensors,
- both runs complete in strict mode, so the MSDA dispatch used the native MUSA operator.

Enable `MUSPACE_DISPATCH_LOG=1` to print every MSDA backend decision. Results and scratch artifacts
belong under `.tmp/` and are not committed.
