# LingBot-VA fixture validation

The recipe exposes one validation entry point with two scopes. `--mode smoke` is
dataset-free and checks the MUSA device layer plus the FlexAttention path the
patches exist for. `--mode train` runs the real post-training entrypoint on a
prepared LeRobot-Latent dataset with the released `lingbot-va-base` weights.
Neither scope is an accuracy or performance benchmark.

## Prerequisites

- A pinned checkout (`scripts/prepare_checkout.sh`) whose `HEAD` equals
  `58c2ae5bac46bd8114065bea9d7d256eb67c16c3`.
- A MUSA device and a stack where `import torchada` succeeds (FlexAttention
  needs the CUDA API compatibility layer).
- For `--mode train`: the latents dataset described in
  [README.md](README.md#prepare-the-dataset) and the `lingbot-va-base`
  transformer weights.

## Commands

Dataset-free smoke, on any MUSA host:

```bash
model_examples/LingBotVA/scripts/prepare_checkout.sh .tmp/LingBotVA
bash model_examples/LingBotVA/scripts/validate_fixture.sh \
  --checkout .tmp/LingBotVA \
  --output-dir .tmp/lingbotva_fixture
```

Real-data validation, single node, 8 cards:

```bash
bash model_examples/LingBotVA/scripts/validate_fixture.sh \
  --checkout .tmp/LingBotVA \
  --mode train --ngpu 8 --num-steps 3 \
  --dataset-path "$LEROBOT_LATENT_ROOT" \
  --pretrained-model-path "$LINGBOT_VA_BASE" \
  --ifname "$RDMA_IFACE" \
  --output-dir .tmp/lingbotva_train
```

Both commands run `mu-space migrate lingbotva` first, then the shared driver
`scripts/train_step_musa.py`, and finally check the emitted JSON. `--mode smoke`
without a checkpoint path and `--mode train` with `--ngpu 1` are equally valid;
the driver records the exact configuration it ran.

## Acceptance contract

The JSON written to `<output-dir>/<mode>.json` must satisfy the common fields
plus the scope-specific ones:

| Field | Requirement |
| --- | --- |
| `device` | `musa:0` |
| `optimizer_step` | `true` - the module or the trainer performed an optimizer update |
| `finite_gradients` | `true` |
| `losses` | non-empty and all values finite |
| `block_mask_gate_all_pass` | smoke only: compiled `create_block_mask(dynamic=True)` passes at every requested length |
| `flex_forward_backward_finite` | smoke only: compiled FlexAttention forward/backward produces finite output and gradients |
| `training_completed` | train only: the trainer logged `Training completed!` |
| `failure_markers` | train only: empty (no `sort.stable`, NaN, `FloatingPointError`, or OOM in the log) |

## Negative control

The block-mask gate is the regression test for the MUSA workaround. On the same
host and stack, an unpatched checkout must fail it:

```bash
model_examples/LingBotVA/scripts/prepare_checkout.sh .tmp/LingBotVA-clean
MUSA_VISIBLE_DEVICES=0 python model_examples/LingBotVA/scripts/train_step_musa.py \
  --checkout .tmp/LingBotVA-clean --mode smoke --output-dir .tmp/lingbotva_clean
```

The gate must report `block_mask_gate_all_pass: false` with an
`aten.sort.stable` stride assertion, while the patched checkout passes. A bare
`torch.compile(torch.argsort, dynamic=True)` on a transposed 4-D input still
fails after patching: the workaround wraps only the FlexAttention
`_dense_to_ordered` helper, it does not change the backend.

## Measured output

Validated on 8x MTT S5000, MUSA 5.2.0, PyTorch `2.11.0.post1`, torch_musa
`2.11.0.post1+b431668`, torchada `0.1.86`, against upstream revision
`58c2ae5bac46bd8114065bea9d7d256eb67c16c3`.

**Patched smoke** (`--mode smoke`, one card):

```text
recipe_patches_present   true
device / device_layer    musa:0 / musa:0
torchada_active          true
dist_backend             nccl (MCCL through torchada)
block_mask_gate          {3968: true, 4096: true, 4224: true, 5504: true}
flex_forward_backward_finite                 true
attention_module_forward_backward_finite     true
optimizer_step / finite_gradients            true / true
losses                   [0.0193...]
```

The same JSON was produced in two independent containers on different hosts
(both with the same MUSA stack), and `validate_fixture.sh` printed
`LingBot-VA smoke validation passed`.

**Negative control** (same host, unpatched checkout):

```text
recipe_patches_present     false
block_mask_gate_all_pass   false
block_mask_gate            {3968: false, 4096: false, 4224: false, 5504: false}
error (length 3968)        AssertionError: expected size 31==31, stride 31==1 at dim=2;
                           expected size 31==31, stride 1==31 at dim=3
                           Error in op: torch.ops.aten.sort.stable
```

**Real-data training** (`--mode train`, 8 cards, `attn_mode=flex`, MBS=1,
gradient accumulation 10, real LeRobot-Latent dataset and `lingbot-va-base`):

| Run | step | latent loss | action loss | step time |
| --- | --- | --- | --- | --- |
| 1 optimizer step | 0 | 0.1561 | 0.4700 | 82.64 s |
| 3 optimizer steps | 0 | 0.1561 | 0.4700 | 81.45 s |
| 3 optimizer steps | 1 | 0.1518 | 0.4653 | 68.13 s |
| 3 optimizer steps | 2 | 0.1506 | 0.4276 | 69.67 s |

Both runs exited 0 and logged `Training completed!` with an empty
`failure_markers` list, and `validate_fixture.sh --mode train` printed
`LingBot-VA train validation passed` for each scope. The 1-step run used
`--save-interval 1` to exercise the recipe's sharded atomic checkpoint publish:
the trainer logged `Checkpoint saved and validated successfully at step 1` and
published `checkpoint_step_1/transformer/` with five safetensors shards plus the
diffusers `config.json`. The loss values
reproduce the recorded single-node baseline exactly (step 0 `latent=0.1561`,
`action=0.4700`), which is the expected outcome of applying the same MUSA patch
set to the same pinned revision.

