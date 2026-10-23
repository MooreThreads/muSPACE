# Radius benchmark

## Scope

Measured entry points, one per case:

- `mu_space.ops.radius` (`av2-enc-pl2a`) — the QCNet scene-encoder polygon-to-agent
  neighborhood query: which agent-step points lie within 50 m of a polygon-step point of the
  same historical step.
- `mu_space.ops.radius_graph` (`av2-enc-a2a`) — the QCNet scene-encoder agent-to-agent graph:
  which agent-step points of the same historical step lie within 50 m of each other.
- `mu_space.qcnet_compat` (`av2-enc-compat`) — the integration path. It times the two queries
  above issued through the device-aware compatibility layer with the QCNet call signature
  (keyword arguments, no `backend` override), so the timed region additionally contains the
  compatibility layer's device check, its argument forwarding for `num_workers` and
  `max_num_neighbors`, and its strict native selection on a MUSA tensor. QCNet issues the two
  queries as one neighborhood-construction step of the scene encoder, so the integration case
  times them together rather than one at a time.

Excluded from the timed region in every case: construction of the fixture from the fixed seed,
the host-to-device transfer of the fixture, the sorted segment-label build, and every consumer
of the returned `[2, E]` edge index. Nothing is added inside the timed region beyond the entry
point's own work: the timed region is `--iterations` calls of one entry point on tensors that
were built and transferred before the first warmup.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `av2-enc-pl2a` | `mu_space.ops.radius`; 1 Argoverse 2 scenario x 50 historical steps x 60 tracked agents = 3000 agent-step points queried by 600000 polygon-step points (12000 map polygon vertices repeated over the steps); 2 coordinate channels; 50 m radius; 300-neighbor cap | `float32` | `2333` |
| `av2-enc-a2a` | `mu_space.ops.radius_graph`; the same scene, 3000 agent-step points in 50 step-isolated segments; `loop=False`; 50 m radius; 300-neighbor cap | `float32` | `2333` |
| `av2-enc-compat` | `mu_space.qcnet_compat`; the same scene queried through the compatibility layer with the QCNet call signature: the polygon-to-agent query and the agent-to-agent graph of one scene-encoder step | `float32` | `2333` |

All three cases build one shared deterministic scene: positions are drawn from the fixed seed
inside a 250 m x 250 m local-map box, segment labels are sorted and non-negative, and every
point count stays inside the int32 indexing range of the native path. The largest neighborhood
family a query point can see is the 60 agents of its own historical step, so the fixture never
saturates the 300-neighbor cap.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

Scope is `operator` for all three cases. The protocol is the default one: `torch.musa.Event`
device events, `--warmup 20`, `--iterations 100`, `--repeats 5`, `--seed 2333`, and the
published latency is the median of the five samples. No deviation from the default protocol is
taken and no wall-clock timer is used.

The fixture is built, transferred to the device and validated before the first warmup launch.
Inside the timed region each iteration calls the entry point once and discards the result, so
the following entry-point work is measured and cannot be separated from the kernels through the
public API:

- the fixed-capacity int64 neighbor buffer the native path allocates and then compacts:
  600000 x 300 slots for `av2-enc-pl2a` (1.341 GiB), 3000 x 301 slots for `av2-enc-a2a`
  (0.007 GiB, one slot per query point reserves the self pair that `loop=False` drops), and
  both for `av2-enc-compat` (1.348 GiB);
- one device-to-host read-back of the compacted edge count, which the entry point uses to size
  the returned edge index;
- for `av2-enc-compat`, both queries of the encoder step, with the compatibility layer's device
  check and argument forwarding around them.

Excluded: fixture construction from the fixed seed, the host-to-device transfer, the sorted
segment-label build, and any downstream use of the edge index. The measured fixture returns
3795384 neighbor pairs for the polygon-to-agent query and 18344 edges for the agent-to-agent
graph, so the fixed-capacity buffers are much larger than the results they hold.

The published S5000 rows use the FP32 latency. `radius` and `radius_graph` accept FP32 only, so
`float32` is the operator's production dtype and the only one measured.

## Run

Run from the repository root on the measured device, in an environment with an importable MUSA
PyTorch stack (`docs/install.md`):

```bash
PYTHONPATH=. python benchmarks/radius/benchmark_radius.py --case av2-enc-pl2a \
  --warmup 20 --iterations 100 --repeats 5 --output .tmp/benchmarks/radius-av2-enc-pl2a.json
PYTHONPATH=. python benchmarks/radius/benchmark_radius.py --case av2-enc-a2a \
  --warmup 20 --iterations 100 --repeats 5 --output .tmp/benchmarks/radius-av2-enc-a2a.json
PYTHONPATH=. python benchmarks/radius/benchmark_radius.py --case av2-enc-compat \
  --warmup 20 --iterations 100 --repeats 5 --output .tmp/benchmarks/radius-av2-enc-compat.json
```

`--op` overrides the entry point registered for a case (`radius`, `radius_graph`,
`qcnet_compat`). It exists for local investigation only: a published row always uses the entry
point registered for its case. The reference side is upstream `torch-cluster`'s `radius` and
`radius_graph`, measured on the reference device with the same case, dtype, seed and protocol;
for `av2-enc-compat` the compatibility layer routes a non-MUSA tensor to that same upstream
implementation, so the integration row can be reproduced through the same entry point.

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `av2-enc-pl2a` | `float32` | `operator` | 2.0548 | 1.6990 | 6.7956 | 4.00x |  |
| `av2-enc-a2a` | `float32` | `operator` | 0.6845 | 0.5660 | 0.3330 | 0.59x |  |
| `av2-enc-compat` | `float32` | `operator` | 2.7450 | 2.2698 | 7.1751 | 3.16x |  |

## Acceptance

Pass criterion: the documented command reproduces the published S5000 p50 for the case under the
published protocol, the value does not regress against the previously accepted S5000 number for the
same case, dtype and scope, and `tests/ops/test_radius.py` passes on the measured build. Every case
also has a published reference row measured on the reference device with the same case, dtype, scope
and protocol, and the comparison against it is reported in the table.

`av2-enc-a2a` is the one published row where the measured device is slower than the reference device
(0.59x). Both sides build the same neighborhood from the same seed and produce the same edge count
(18,344), and each times exactly one entry-point call, so the row is a real measurement rather than a
boundary artefact: at that size the timed region is dominated by the fixed per-call work of the entry
point, not by neighbor-search throughput. Closing that gap is outstanding optimization work, and the
row stays published with its number.

## Notes

- These are entry-point measurements. The fixed-capacity neighbor-buffer allocation and the
  single device-to-host read-back described in [Measurement](#measurement) are inside the timed
  region because they are inside the public entry point, so these rows must not be compared with
  a bare-kernel figure for the same shapes, and a reader must not read them as kernel-only
  latency.
- Scope is `operator` for all three cases, including the two primitives: one call of a public
  entry point is the boundary a caller sees. Rows with different scopes are never merged.
- Every row has a published reference row measured on the reference device with the same case,
  dtype, scope and protocol: upstream `torch-cluster`'s `radius` and `radius_graph`. The muSPACE
  PyTorch fallback (`radius_reference`, `radius_graph_reference`) is a correctness path and never
  a reference platform.
- The published coordinate dimension is 2, the ground-plane value QCNet queries. The operator's
  documented contract is `D < 9`; the wider coordinate dimensions are established by
  `tests/ops/test_radius.py`, not by these rows.
- The modeled context is the QCNet scene encoder of the Argoverse 2 motion-forecasting recipe:
  50 historical steps, 60 tracked agents per scenario, a 50 m query radius and a 300-neighbor
  cap, all taken from that recipe. The recipe contains no performance validation of its own, so
  these rows are the operator-level evidence for it and not an end-to-end model claim.
- Latency is dominated by two different costs depending on the case: `av2-enc-pl2a` and
  `av2-enc-compat` are dominated by the large fixed-capacity buffer, while `av2-enc-a2a` has a
  0.007 GiB buffer and is dominated by the entry point's fixed per-call work. The comparison
  with the reference implementation is what decides the acceptance criterion in both cases.
