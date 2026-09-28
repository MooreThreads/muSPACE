# OpenVLA fixture validation

This fixture follows the BEVFormer model-example validation contract while keeping the OpenVLA
scope intentionally small: one MUSA card, synthetic image/text/action-token inputs, three FP32
optimizer steps, and a checkpoint under `.tmp/`.

## Command

Prepare and patch the pinned official checkout, then run the validation entry point:

```bash
model_examples/OpenVLA/scripts/prepare_checkout.sh .tmp/OpenVLA
bash model_examples/OpenVLA/scripts/validate_fixture.sh \
  --checkout .tmp/OpenVLA \
  --output-dir .tmp/openvla_fixture
```

`validate_fixture.sh` applies the manifest patch through `mu-space migrate`, launches
`train_step_musa.py --mode fixture`, writes `default.json` and `tiny_openvla_step.pt`, then checks
the JSON contract. The required fields are `mode=fixture`, `device=musa:0`, finite `loss`/`losses`,
`finite_gradients=true`, and `optimizer_step=true`.

Use `--device musa:0` for an explicit device selection. It does not download the 7B checkpoint
or run the full RLDS data pipeline.
