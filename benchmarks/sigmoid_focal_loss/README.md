# Sigmoid focal loss benchmark

## Scope

Measured: `mu_space.ops.sigmoid_focal_loss` through its public autograd boundary, with `alpha=0.25`,
`gamma=2.0`, `reduction="mean"` and `backend="auto"` — the loss contract of the BEVFusion
`BEVSegmentationHead`. `--case` selects the workload and the direction.

The reference implementation of this operator on the reference device is
`torchvision.ops.sigmoid_focal_loss`, a fused CUDA operation with its own fused backward; it is fed
the case's inputs and timed under the same protocol as the published rows. The in-repo PyTorch
expression the runner evaluates with `--device cuda` is a cross-check of the case, not the source
of a published reference latency.

Outside the timed region: case construction, host-to-device transfer of the case tensors, and the
dispatch report. A forward row times one operator call. A backward row times one operator call plus
the backward pass it triggers, which is how the operator's autograd boundary exposes the backward
direction.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `bevfusion-seg-forward` | logits `[4, 3, 200, 200]`, binary targets of the same shape, `alpha 0.25`, `gamma 2`, `reduction mean`; forward call only | `float32` | `2333` |
| `bevfusion-seg-backward` | the `bevfusion-seg-forward` tensors, with the logits as the differentiated leaf; times the forward call plus the backward pass it triggers | `float32` | `2333` |
| `small-forward` | logits and targets `[2, 3, 64, 64]`, `alpha 0.25`, `gamma 2`, `reduction mean`; forward call only | `float32` | `2333` |
| `small-backward` | the `small-forward` tensors, with the logits as the differentiated leaf; times the forward call plus the backward pass it triggers | `float32` | `2333` |
| `wide-forward` | logits and targets `[8, 256, 1]`, `alpha 0.25`, `gamma 2`, `reduction mean`; forward call only | `float32` | `2333` |
| `wide-backward` | the `wide-forward` tensors, with the logits as the differentiated leaf; times the forward call plus the backward pass it triggers | `float32` | `2333` |

Logits are standard normal and targets are binary; both are drawn in that order from the seed's single
stream and cast to the measured dtype after transfer, so the same sample underlies every dtype.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

Every row uses `operator` scope: the timed region covers one direction of the operator boundary and
nothing else. Each sample records one device event pair around `--iterations` launches and divides the
elapsed time by `--iterations`; warmup launches are untimed; the published number is the median of
`--repeats` samples. Wall-clock timing is not used.

Preallocated before the timed region: the logits and targets on the measured device. The logits leaf
is derived once with `detach().requires_grad_(True)`; the backward rows clear `.grad` and run one
forward and one backward per launch inside the timed region, so gradients never accumulate across
launches.

The default protocol is used unchanged: `--warmup 20 --iterations 100 --repeats 5`, seed `2333`.

Each case is published as the median over five full executions of the documented command, on both
devices, per the `Executions` rule of `docs/benchmarking.md`: both records of a pair state
`"executions": 5` and list every per-execution median. The reference rows are measured on the
reference device with the fused `torchvision.ops.sigmoid_focal_loss` CUDA operator at the same
boundary - one fused forward, or one fused forward plus one backward per timed launch - because
the operator's plain PyTorch expression is a chain of small kernels whose latency in this
container is dominated by host-side dispatch rather than by device work; that expression is used
only as the numerical cross-check of the fused reference. The five S5000 executions span
0.0440-0.0509 ms on the forward rows and 0.1463-0.1819 ms on the backward rows. The five reference
executions span 0.1546-0.1819 ms on the forward rows and 0.4381-1.4495 ms on the backward rows,
whose fused backward still carries its own autograd dispatch; the record lists the per-execution
medians that bound both bands.

## Run

```bash
PYTHONPATH=. python benchmarks/sigmoid_focal_loss/benchmark_sigmoid_focal_loss.py \
  --case bevfusion-seg-forward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/sigmoid_focal_loss-bevfusion-seg-forward.json
PYTHONPATH=. python benchmarks/sigmoid_focal_loss/benchmark_sigmoid_focal_loss.py \
  --case bevfusion-seg-backward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/sigmoid_focal_loss-bevfusion-seg-backward.json
PYTHONPATH=. python benchmarks/sigmoid_focal_loss/benchmark_sigmoid_focal_loss.py \
  --case small-forward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/sigmoid_focal_loss-small-forward.json
PYTHONPATH=. python benchmarks/sigmoid_focal_loss/benchmark_sigmoid_focal_loss.py \
  --case small-backward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/sigmoid_focal_loss-small-backward.json
PYTHONPATH=. python benchmarks/sigmoid_focal_loss/benchmark_sigmoid_focal_loss.py \
  --case wide-forward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/sigmoid_focal_loss-wide-forward.json
PYTHONPATH=. python benchmarks/sigmoid_focal_loss/benchmark_sigmoid_focal_loss.py \
  --case wide-backward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/sigmoid_focal_loss-wide-backward.json
```

A reference row is measured on the reference device by feeding the same case inputs, built by the
same case module, to the fused upstream operation and timing that call with the same harness,
protocol and execution count as the measured row. The `--device cuda` invocation of the runner
above only cross-checks the case; it does not produce a published reference latency.

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `bevfusion-seg-forward` | `float32` | `operator` | 0.0495 | 0.0409 | 0.1659 | 4.06x |  |
| `bevfusion-seg-backward` | `float32` | `operator` | 0.1590 | 0.1315 | 0.8154 | 6.20x |  |
| `small-forward` | `float32` | `operator` | 0.0453 | 0.0374 | 0.1554 | 4.15x |  |
| `small-backward` | `float32` | `operator` | 0.1525 | 0.1261 | 0.5822 | 4.62x |  |
| `wide-forward` | `float32` | `operator` | 0.0452 | 0.0374 | 0.1581 | 4.23x |  |
| `wide-backward` | `float32` | `operator` | 0.1664 | 0.1376 | 0.6957 | 5.06x |  |

## Acceptance

`pytest tests/ops/test_sigmoid_focal_loss.py -q` passes on the measured build, and five executions
of the documented command reproduce the published S5000 p50 as the median of the five
per-execution medians listed in the record. A re-measurement whose median falls outside the
per-execution range recorded for that row re-opens the row.

## Notes

- The reference rows are measured with the fused `torchvision.ops.sigmoid_focal_loss` CUDA
  implementation, which reproduces this operator's contract (`alpha=0.25`, `gamma=2.0`,
  `reduction="mean"`, binary per-element targets) to `1.2e-08` on the forward loss and `8.8e-12` on
  the gradient at `float32`. `mmcv.ops.sigmoid_focal_loss` was considered and rejected: it
  implements the multi-class class-index contract (`input` of shape `[N, C]`, one `int64` label per
  row, mean over rows), so it neither accepts the case tensors nor computes the same loss.
- The `S5000 normalized (ms)` cell adjusts the measured latency by nominal device bandwidth: the
  S5000 p50 multiplied by `1600 / 1935`, the registered bandwidths of the measured and the reference
  device. `Speedup` is the reference p50 divided by that normalized latency; every row is rendered
  from its paired records with `python benchmarks/_harness.py row --s5000 <record> --reference
  <record>`.
- The forward rows are shaped like production inference and training grids, but at these shapes the
  latency is dominated by launch and dispatch overhead rather than by memory traffic: the three
  forward rows differ by less than `13%` although their element counts differ by more than two orders
  of magnitude. Read them as the cost of the operator boundary, not as a throughput measurement.
- `small` and `wide` are shape sanity rows that keep the boundary honest at other grids; the
  production workload is `bevfusion-seg`.
- The runner and the operator also accept `float16` logits and targets. Only `float32` rows are
  published, per the fixed protocol; a `float16` row would need its own reference measurement.
