# QCNet Real-Subset Validation

The fixture runs the official QCNet training step directly on real Argoverse 2 data. The default contract is
one S5000 card, batch size one, one optimizer step, 50 historical steps, 60 future steps, and the
upstream model dimensions and neighborhood radii.

## Acceptance criteria

The run passes only when:

- the checkout matches the pinned commit and the QCNet migration is applied;
- the device is MUSA and `radius`/`radius_graph` select the native backend;
- training completes the requested number of optimizer steps;
- all resulting model parameters are finite;
- all gradients used for the update are finite;
- the real batch, initial state, and trained state artifacts are written.

Run FP32 and FP16 separately so a half-precision failure cannot be hidden by an FP32 pass. These
checks prove launchability and a bounded optimizer update, not convergence or quality.

## Precision artifacts

`precision_compare.py` consumes the saved CPU batch and initial state. It evaluates the same model
with dropout disabled and stores all tensor-valued QCNet outputs plus training loss as FP32 result
artifacts. A comparison fails if keys or shapes differ, any output is non-finite, or any element is
outside the selected tolerance.

## Troubleshooting

- An import failure for `torch_geometric`, Lightning, AV2, or a scatter dependency indicates an
  incomplete model environment, not a radius mismatch.
- A native radius error means the installed muSPACE wheel does not match the source checkout or the
  MUSA device is unavailable; rebuild and reinstall the wheel before retrying.
- A CUDA torch-cluster import or device error invalidates the A100 baseline. Do not substitute the
  muSPACE reference implementation.
- Preserve raw JSON metrics while investigating a mismatch, but do not publish an alignment claim
  until both dtypes meet their independent thresholds.
