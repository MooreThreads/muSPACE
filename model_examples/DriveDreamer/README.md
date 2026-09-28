# DriveDreamer for muSPACE

This recipe runs the official DriveDreamer image-stage training graph at
`da1ca92f831bc23d91b59ad418eb47b41cbb1fa9`.  It provides both a real-data
single-card trainer launch and a deterministic one-sample numerical-regression check.
The trainer launch is the official multi-batch flow; the regression check is intentionally
bounded to one optimizer step.  Neither path claims video-stage quality or throughput.

DriveDreamer does not bind any of the model-specific operators currently replaced by
muSPACE.  `torch_musa` executes its ordinary PyTorch/Diffusers graph on MUSA.  The
recipe deliberately installs no unrelated muSPACE patcher; its result records that
fact instead of claiming native muSPACE-operator dispatch.

## Prerequisites

Run inside a MUSA development container with matching PyTorch and `torch_musa`.
Install the recipe dependencies into that environment, preserving the container's
existing Torch packages:

```bash
python -m pip install -r model_examples/DriveDreamer/requirements.lock
```

Do not use `--force-reinstall` or install a PyPI `torch` wheel in this container:
the preinstalled, SDK-matched `torch` and `torch_musa` pair must remain intact.
The upstream HDMap converter requires Shapely 1.x; this recipe pins that constraint.

Obtain nuScenes mini under its license and a local copy of
`runwayml/stable-diffusion-v1-5` under its model terms.  The converter requires the
raw mini root to contain `v1.0-mini/` and `samples/`.  All data and artifact paths
below are examples; keep them outside tracked source.

## Prepare

Verify the official checkout, then convert mini and validate or download the local
Stable Diffusion files:

```bash
model_examples/DriveDreamer/scripts/prepare_checkout.sh ./DriveDreamer
bash model_examples/DriveDreamer/scripts/prepare_assets.sh \
  --checkout ./DriveDreamer --raw-nuscenes <raw-nuscenes-root> \
  --output-root <converted-output-root> --models-dir <models-root> \
  --download-sd15
```

If the models were downloaded separately, omit `--download-sd15`.  The model root
must contain `runwayml/stable-diffusion-v1-5/{vae,unet,scheduler}` and
`openai/clip-vit-large-patch14`; the latter is used by the official box-label transform.

## MUSA one-step launch

```bash
bash model_examples/DriveDreamer/scripts/train_1p_image_smoke.sh \
  --checkout ./DriveDreamer --data-root <converted-cam-all-train> \
  --models-dir <models-root> --input-artifact <input-artifact> \
  --artifact <musa-artifact> --device musa --sample-index 0
```

The first invocation writes the portable input artifact.  It includes the actual
collated sample, initialization state, VAE sample noise, diffusion noise, and
timesteps.  Later invocations replay it and therefore do not reread the dataset.
The recipe explicitly disables CUDA/MUSA matmul and cuDNN/muDNN TF32 controls where
they are exposed by the active PyTorch backend.  A successful result has finite loss
and gradient, plus a changed DriveDreamer parameter.

## Full image-stage training launch

This command calls the upstream `DriveDreamerTrainer` with its complete dataloader,
checkpoint, logging, scheduler, and epoch loop. Paths are injected at runtime; the
upstream hard-coded `ENV.py` and config are not edited.

```bash
bash model_examples/DriveDreamer/scripts/train_full_musa.sh \
  --checkout ./DriveDreamer \
  --data-root <converted-cam-all-train> \
  --models-dir <models-root> \
  --output-dir <training-output> \
  --epochs 1 --gpu-id 0
```

`--epochs 1` is a launch-validation setting; increase it for an actual training run.
This path reads all samples through the official dataloader and writes checkpoints and
TensorBoard logs under `--output-dir`.

## Real-data validation

Use the one-command validator to exercise a real transformed sample and check finite
loss, backward, and an optimizer update. It does not replace the full trainer launch.

```bash
bash model_examples/DriveDreamer/scripts/validate_real.sh \
  --checkout ./DriveDreamer \
  --data-root <converted-cam-all-train> \
  --models-dir <models-root> \
  --input-artifact <input-artifact> \
  --artifact <musa-artifact> --device musa
```

## A100 alignment

Copy the unchanged upstream checkout, model files, and portable input artifact to a
verified A100 container.  Run the same launch command with `--device cuda` and a
distinct output artifact.  Use a local, non-submitted validation harness to compare
the stage tensors and report the first failing stage; a matching final loss alone is
not sufficient evidence of alignment.
