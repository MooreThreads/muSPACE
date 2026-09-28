# LingBot-VA

## Contents

- [Overview](#overview)
  - [Model description](#model-description)
  - [Code implementation](#code-implementation)
- [Preparing the environment](#preparing-the-environment)
  - [Install dependencies](#install-dependencies)
  - [Prepare the checkout](#prepare-the-checkout)
  - [Prepare the dataset](#prepare-the-dataset)
  - [Prepare pretrained weights](#prepare-pretrained-weights)
- [Quick start](#quick-start)
  - [Runtime patching](#runtime-patching)
  - [Dataset-free smoke](#dataset-free-smoke)
  - [Post-training launch](#post-training-launch)
- [Validation](#validation)
- [Results](#results)
- [Known limitations](#known-limitations)
- [Changelog](#changelog)

# Overview

## Model description

| Component | Requirement |
| --- | --- |
| Python | `3.10` |
| PyTorch | `2.11.0.post1` |
| torch_musa | `2.11.0.post1` (matching PyTorch) |
| torchada | `0.1.86` |
| MUSA Toolkit | `5.2.0` |
| Device | MTT S5000 |
| LingBot-VA | `58c2ae5bac46bd8114065bea9d7d256eb67c16c3` |

CuDNN/CUDA-only dependency stacks are not required. `mu-space env` reports the
active stack; `torchada` must be importable for FlexAttention to work on MUSA.

LingBot-VA is an autoregressive video-action world model. A 30-layer
Mixture-of-Transformers backbone carries two streams in one interleaved causal
sequence: a wide video stream that predicts future visual latents with flow
matching, and a narrow action stream that inverts visual transitions into a
30-D action chunk. Training runs the two streams jointly, so self-attention sees
video and action tokens together with a token-level visibility rule
(`seq_id`, frame distance window, clean/noise causal ordering) expressed as a
FlexAttention `BlockMask`, while text conditioning uses a cross-attention mask.
Sequence length is data dependent (~3968-5504 tokens for the LIBERO-LONG
post-training set), which is why the BlockMask path dominates MUSA integration.

## Code implementation

- Reference implementation:

  ```shell
  url=https://github.com/Robbyant/lingbot-va
  commit_id=58c2ae5bac46bd8114065bea9d7d256eb67c16c3
  ```

- muSPACE recipe:

  ```shell
  code_path=model_examples/LingBotVA
  ```

muSPACE maintains no model fork. The recipe pins the upstream revision, applies
four compatibility patches under `patches/`, and drives the official trainer
either with a dataset-free smoke or with real post-training data.

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Pinned upstream revision, patch metadata, and recipe entrypoints |
| `patches/` | MUSA device runtime, attention/FlexAttention compatibility, training entrypoint, fused RoPE and FSDP tuning |
| `scripts/prepare_checkout.sh` | Fetch and verify the pinned official checkout |
| `scripts/validate_fixture.sh` | Apply the patches and run the smoke or real-data contract |
| `scripts/train_step_musa.py` | Shared MUSA driver for `--mode smoke` and `--mode train` |
| `scripts/train_posttrain_smoke.sh` | Real-data post-training launcher (dataset + weights required) |
| `fixture_validation.md` | Reproducible commands and the acceptance contract |
| `requirements.lock` | Supported dependency scope for the MUSA recipe |

The patches are:

| Patch | Scope |
| --- | --- |
| `lingbot-va-01-musa-runtime.patch` | `wan_va/utils/device.py` and `wan_va/utils/musa_bootstrap.py`, distributed backend selection, device/cache/synchronize call sites, `FusedAdamW` on MUSA |
| `lingbot-va-02-musa-attention-flex.patch` | Dense-mask `torch`/`flashattn` attention paths, FlexAttention compile gating, and the `_dense_to_ordered` contiguous graph-break that keeps compiled `create_block_mask(dynamic=True)` away from the MUSA `aten.sort.stable` stride assertion |
| `lingbot-va-03-training-entrypoint.patch` | CLI overrides, seeded runs, `drop_last`, non-finite loss/gradient guards, step-time logging, and sharded atomic checkpoint publishing |
| `lingbot-va-04-rope-fusion-and-fsdp-tuning.patch` | `wan_va/modules/rope_fused.py`, a single-kernel 3D RoPE that stays in the autograd graph, plus the FSDP sharding-granularity and reshard toggles in `wan_va/distributed/fsdp.py` |

## Preparing the environment

### Install dependencies

Build and install the current muSPACE wheel so `mu-space migrate` matches the
checkout, then install the recipe dependencies:

```bash
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir .tmp/wheelhouse
python -m pip install --force-reinstall --no-deps .tmp/wheelhouse/mu_space-*.whl
python -m pip install torchada==0.1.86 diffusers==0.36.0 einops easydict websockets msgpack
python -m pip install lerobot==0.3.3   # only for the real-data training mode
```

`requirements.lock` documents the supported scope; `torch`/`torch_musa` come
from the MUSA image and must stay at matching versions.

### Prepare the checkout

```bash
model_examples/LingBotVA/scripts/prepare_checkout.sh .tmp/LingBotVA
```

The script verifies `origin`, fetches the pinned revision, and confirms
`wan_va/train.py`. On a host without access to the upstream repository, prepare
the checkout elsewhere and copy it in; `mu-space migrate` only needs a git
checkout whose `HEAD` equals the pinned revision.

### Prepare the dataset

`--mode train` consumes a LeRobot-format latent dataset produced by the upstream
data pipeline. Each episode stores pre-computed Wan2.2 VAE latents next to the
aligned 30-D actions, plus the text embedding of the instruction:

```text
<dataset-root>/
├── meta/info.json               # LeRobot metadata
├── meta/episodes.jsonl          # episode list and action descriptions
├── latents/chunk-000/<camera>/episode_000000_<start>_<end>.pth
├── empty_emb.pt                 # unconditioned embedding for CFG dropout
└── (videos/ is not read during post-training)
```

The loader walks `<dataset-root>` for `meta/info.json`, so any directory layout
that preserves that invariant works. Keep the raw dataset read-only: the recipe
never writes into it.

### Prepare pretrained weights

Post-training starts from the released `lingbot-va-base` weights. Point
`--pretrained-model-path` at a directory containing the `transformer/`
subfolder with the diffusers-format `config.json` and safetensors shards.
The VAE and text encoder are not loaded by the training entrypoint, so only the
transformer weights are needed.

## Quick start

### Runtime patching

```bash
python -m mu_space migrate lingbotva --checkout .tmp/LingBotVA
```

`migrate` applies the four patches in order and records
`.mu_space_lingbotva.json`. `python -m mu_space restore lingbotva --checkout .tmp/LingBotVA`
reverses the recipe.

### Dataset-free smoke

Any MUSA host with the patched checkout, without dataset or weights:

```bash
bash model_examples/LingBotVA/scripts/validate_fixture.sh \
  --checkout .tmp/LingBotVA \
  --output-dir .tmp/lingbotva_fixture
```

The smoke reports the device layer, compiles `create_block_mask(dynamic=True)`
at LIBERO-realistic sequence lengths, runs a compiled FlexAttention
forward/backward, and takes one optimizer step on a small attention module.
Run it on an unpatched checkout to see the negative control: the block-mask gate
fails with the `aten.sort.stable` stride assertion.

### Post-training launch

```bash
DATASET_PATH=<LEROBOT_LATENT_ROOT> \
PRETRAINED_MODEL_PATH=<LINGBOT_VA_BASE> \
NGPU=8 NUM_STEPS=1 SAVE_ROOT=<OUTPUT_ROOT> \
IFNAME=<RDMA_IFACE> \
bash model_examples/LingBotVA/scripts/train_posttrain_smoke.sh
```

The launcher runs `python -m torch.distributed.run -m train --config-name libero_train`
with the validated single-node configuration: one sample per rank per
micro-batch, `GRADIENT_ACCUMULATION_STEPS` micro-batches per optimizer step,
`attn_mode=flex`, and `LINGBOT_SEED` fixed. The launcher runs `mu-space migrate`
first, so the checkout carries the recipe patches.

Environment variables understood by the driver:

| Variable | Meaning |
| --- | --- |
| `LINGBOT_DEVICE=musa` | Force the device layer (`musa`, `cuda`, or `cpu`) |
| `LINGBOT_FLEX_SORT_CONTIGUOUS=1` | Enable the Flex block-mask workaround (default for `attn_mode=flex`) |
| `LINGBOT_FLEX_DYNAMIC=1` | Compile the main FlexAttention operator with `dynamic=True` |
| `LINGBOT_ROPE_MODE=kernel` | Apply 3D RoPE with the fused Triton kernel; default on both attention paths, `eager` restores the upstream fp64 complex-multiply chain |
| `LINGBOT_FSDP_SHARD_GRANULARITY=block` | One FSDP unit per transformer block instead of one per attention/FFN submodule; `unit` is the upstream layout, `model` shards the root only. Defaults to `block` for `attn_mode=flashattn` and to `unit` for `attn_mode=flex` |
| `LINGBOT_FSDP_RESHARD_AFTER_FORWARD=0` | Keep the parameters resident between the forward and the backward pass instead of all-gathering them twice. Defaults to `0` for `attn_mode=flashattn` and to `1` for `attn_mode=flex`, where the extra residency does not fit |
| `LINGBOT_DISABLE_FLEX=1` | Fall back to dense `torch` attention |
| `LINGBOT_SEED` | Seed set as `seed + rank` |
| `LINGBOT_CHECK_FINITE` | Not used by this recipe; non-finite losses and gradients fail the run by default |

## Validation

`validate_fixture.sh` accepts two scopes:

| Scope | Requires | Checks |
| --- | --- | --- |
| `--mode smoke` | patched checkout only | device layer reported, compiled BlockMask gate passes at every length, compiled FlexAttention forward/backward finite, one optimizer step on the attention module |
| `--mode train` | dataset + `lingbot-va-base` | training completes, losses finite, no `sort.stable`/NaN/OOM markers, at least one optimizer step |

Both scopes require `optimizer_step: true`, `finite_gradients: true`, and
finite `losses` in the emitted JSON. The real-data scope additionally requires
`training_completed: true` and an empty `failure_markers` list. The full
contract, the negative control, and the measured output are recorded in
[fixture_validation.md](fixture_validation.md).

## Results

Measured on 8x MTT S5000, MUSA 5.2.0, PyTorch `2.11.0.post1`,
torch_musa `2.11.0.post1`, torchada `0.1.86`, LIBERO-LONG post-training data,
one sample per rank, gradient accumulation 10, fixed seed, 20 optimizer steps
with the profiler disabled. Step time is the mean of steps 2-19 (the first two
steps include compilation).

| Attention path | 20-step step time | Loss trajectory (action) | Notes |
| --- | --- | --- | --- |
| FlexAttention, compiled BlockMask (recipe default) | **75.56 s** (p50 73.49, min 69.12, max 94.26) | 0.4700 -> 0.1721 | `LINGBOT_FLEX_SORT_CONTIGUOUS=1`; no `sort.stable`, no NaN |
| FlexAttention, eager BlockMask | 75.74 s | identical to the row above | `LINGBOT_DISABLE_FLEX_MASK_COMPILE=1`; diagnostic comparison |
| Flash SDPA with dense boolean mask | **24.35 s** (3.10x faster) | 0.4701 -> 0.1722 | `attn_mode=flashattn`; MUSA-only path, conditionally substitutable |

Operator paths observed in profiler traces: FlexAttention compiles to Triton
`triton_tem_fused_flex_attention_*` kernels on MUSA, while `attn_mode=flashattn`
reaches the MUSA fused kernel `musa_asm_bf16tce_flash_atten_*_boolmask`. The
upstream dense-mask flash path is a MUSA extension: CUDA PyTorch
`SDPBackend.FLASH_ATTENTION` rejects a non-null `attn_mask`, and CUDA
FlexAttention dispatches to Triton rather than to FlashAttention.

Functional equivalence of the two paths was checked on the CPU with a synthetic
`N=77` layout: visibility of every valid token matches exactly (sequence,
clean/noise causal ordering, and window rules). Only padded rows differ, because
the dense mask forces the padded diagonal visible to avoid softmax NaN; on the
training layout (including padding) the output cosine similarity is 0.817, so
Flash SDPA is a **conditionally substitutable** performance comparison, not a
drop-in replacement. Keep `attn_mode=flex` for customer parity.

### Fused RoPE and FSDP tuning

The two optimizations in `lingbot-va-04-rope-fusion-and-fsdp-tuning.patch` were
measured on the same stack and data over 500 optimizer steps, reporting the mean
of steps 2-499. A 20-step mean carries about +/-1.3% of run-to-run noise because
the data pipeline is not reproducible across launches, so 500 steps is the
shortest span that separates a real effect from that noise.

| Configuration | 500-step step time | Action loss, steps 400-499 |
| --- | --- | --- |
| Flash SDPA, upstream FSDP layout, eager RoPE | 24.56 s | 0.0760 |
| Flash SDPA, fused RoPE, block sharding, no reshard | **19.48 s** (-20.7%) | 0.0762 |

Both rows were run with the same seed, so the loss column is comparable step by
step: the mean absolute per-step difference is 3.0e-4 and the largest is 2.4e-3,
and the gap does not grow over the run. The window mean moves by +0.11% to
+0.28% across the five 100-step windows, which is inside the noise band, so the
change is a pure step-time win.

The `attn_mode=flex` path takes the fused RoPE only. Its FSDP layout is
unchanged, and its 75.56 s figure above was **not** re-measured with the fused
kernel, so treat the flex step time as an upper bound.

### Recipe validation

The patched checkout in this recipe was re-validated against the same stack and
data, and reproduces the recorded trajectory:

| Scope | Result |
| --- | --- |
| dataset-free smoke (`--mode smoke`) | device layer, compiled BlockMask gate at 3968/4096/4224/5504, compiled FlexAttention forward/backward, and one optimizer step on the attention module all pass in two independent containers |
| negative control (unpatched checkout) | the same gate fails every length with the MUSA `aten.sort.stable` stride assertion, so the gate is a real regression test |
| real post-training, 8 cards, 1 step | `latent=0.1561`, `action=0.4700`, step time 82.64 s, checkpoint published and validated (`--save-interval 1`) |
| real post-training, 8 cards, 3 steps | step 0 `0.1561 / 0.4700` (81.45 s), step 1 `0.1518 / 0.4653` (68.13 s), step 2 `0.1506 / 0.4276` (69.67 s), `Training completed!` |

Exact fields, commands, and the negative-control output are recorded in
[fixture_validation.md](fixture_validation.md).

## Known limitations

- The compiled BlockMask workaround is required on MUSA: without it,
  `torch.compile(create_block_mask, dynamic=True)` fails with an
  `aten.sort.stable` stride assertion. The underlying Inductor behaviour (a
  transposed 4-D `argsort` losing its stride assumption) still reproduces on a
  bare `torch.compile(argsort, dynamic=True)` call; report it to the backend.
- The recipe intentionally excludes local diagnostics and rolled-back
  experiments: per-tensor finiteness probes, profiler hooks, an fp32 RoPE
  variant, FSDP prefetch toggles, and length bucketing helpers. They changed no
  validated result and are not part of the supported configuration. The fused
  RoPE kernel and the FSDP sharding toggles in `lingbot-va-04` are separate from
  those and are part of the supported configuration.
- `LINGBOT_FSDP_RESHARD_AFTER_FORWARD=0` and block-level sharding buy their
  step-time saving with peak memory. They fit at one sample per rank with
  gradient accumulation 10 and the LIBERO-LONG sequence lengths, and were fitted
  against an 80 GB card. Re-check peak memory before raising the batch size, the
  sequence length, or the accumulation count.
- The fused RoPE needs a Triton build for the MUSA target. If the kernel cannot
  be built or launched, the call falls back to the eager fp64 chain and warns
  once, so the training result stays valid but the step time returns to the
  untuned figure.
- The dataset-free smoke builds `WanAttention` with `rotary_emb=None`, so it
  never enters the RoPE path and does not compile the fused kernel. The numeric
  evidence for that path is the operator-level comparison against the fp64 chain
  (forward 99.997% of elements bit-equal at the LIBERO-LONG sequence lengths,
  backward cosine similarity 0.99999994) plus the 500-step training comparison
  above. Extending the smoke to cover it is a known gap.
- Multi-node runs are untested; the recipe was validated on a single node.
- FlashAttention mode is a MUSA performance comparison only. The upstream
  `attn_mode=torch` and `flashattn` inference paths are unchanged.
- Only the transformer is trained. VAE and text encoder weights are not part of
  the post-training entrypoint.

## Changelog

- Initial recipe: pinned upstream revision, three MUSA patches, dataset-free
  smoke and real-data post-training entrypoints, and the 8x MTT S5000 results
  above.
- Added `lingbot-va-04-rope-fusion-and-fsdp-tuning.patch`: a single-kernel 3D
  RoPE that stays in the autograd graph, plus FSDP sharding-granularity and
  reshard toggles. Over a 500-step run this cuts the Flash SDPA step time from
  24.56 s to 19.48 s (-20.7%) with loss equal to the untuned run step by step.
