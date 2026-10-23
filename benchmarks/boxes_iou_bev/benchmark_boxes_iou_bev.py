#!/usr/bin/env python3
"""Benchmark the BEVFusion BEV IoU and NMS kernels on MUSA.

``--case`` names the workload and the entry point it belongs to; ``--op`` selects the entry point
explicitly and must agree with the case.  The published protocol and the record format live in
``benchmarks/_harness.py`` and ``docs/benchmarking.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Keep the harness and the local case module importable when the runner is executed directly, as
# the operator README documents.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import add_common_args, make_record, measure, print_summary, resolve_output
from boxes_iou_bev_case import (
    DEFAULT_CASE,
    ENTRY_POINTS,
    FORMAT,
    case_choices,
    make_named_case,
)

OPERATOR = "boxes_iou_bev"
SCOPE = "kernel-only"
NMS_THRESHOLD = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        cases=case_choices(),
        default_case=DEFAULT_CASE,
        dtypes=("float32",),
    )
    parser.add_argument(
        "--op",
        choices=ENTRY_POINTS,
        default=None,
        help="entry point whose kernel is measured; defaults to the entry point of the case",
    )
    parser.add_argument(
        "--num-boxes",
        type=int,
        default=0,
        help="override the box count of the selected case for local investigation",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the complete public API with the independent CPU reference before timing",
    )
    return parser.parse_args(argv)


def build_launch(op: str, boxes: torch.Tensor):
    """Return the timed launcher: one native kernel call, with nothing else inside.

    Every buffer the kernel writes into is allocated here, before the timed region.  The NMS entry
    points launch the bitmask kernel only: the scores are not sorted, the boxes are not gathered,
    and the host-side mask reduction and keep-index compaction are excluded.
    """

    from mu_space.runtime.native import native_module_proxy

    native = native_module_proxy()
    if op == "boxes_iou_bev":
        output = torch.empty(
            (boxes.size(0), boxes.size(0)), dtype=torch.float32, device=boxes.device
        )

        def launch() -> None:
            native.boxes_iou_bev_forward(boxes, boxes, output)

        return launch
    if op not in ("nms_bev", "nms_bev_normal"):
        raise SystemExit(f"unsupported --op {op!r}")
    mask = torch.empty(
        (boxes.size(0), (boxes.size(0) + 63) // 64), dtype=torch.int64, device=boxes.device
    )
    launcher = (
        native.nms_bev_mask_forward if op == "nms_bev" else native.nms_bev_normal_mask_forward
    )

    def launch() -> None:
        launcher(boxes, NMS_THRESHOLD, mask)

    return launch


def check_case(op: str, case: dict[str, object], device: str) -> dict[str, object]:
    """Compare the selected device path with the independent CPU reference."""

    cpu_boxes, cpu_scores = case["boxes"], case["scores"]
    device_boxes, device_scores = cpu_boxes.to(device), cpu_scores.to(device)
    from mu_space import boxes_iou_bev, nms_bev, nms_bev_normal

    if op == "boxes_iou_bev":
        expected = boxes_iou_bev(cpu_boxes, cpu_boxes, backend="reference")
        actual = boxes_iou_bev(device_boxes, device_boxes, backend=device).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        return {"checked": True, "max_abs_error": float((actual - expected).abs().max())}
    if op == "nms_bev":
        expected = nms_bev(cpu_boxes, cpu_scores, NMS_THRESHOLD, backend="reference")
        actual = nms_bev(device_boxes, device_scores, NMS_THRESHOLD, backend=device).cpu()
    else:
        expected = nms_bev_normal(cpu_boxes, cpu_scores, NMS_THRESHOLD, backend="reference")
        actual = nms_bev_normal(device_boxes, device_scores, NMS_THRESHOLD, backend=device).cpu()
    torch.testing.assert_close(actual, expected)
    return {"checked": True, "index_equal": True}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise SystemExit("MUSA is unavailable")
    fixture = make_named_case(args.case, seed=args.seed, op=args.op, num_boxes=args.num_boxes)
    if fixture["format"] != FORMAT:
        raise ValueError("invalid fixture format")
    op = str(fixture["op"])
    check = check_case(op, fixture, args.device) if args.check else {"checked": False}
    boxes = fixture["boxes"].to(args.device)
    launch = build_launch(op, boxes)
    measurement = measure(
        launch,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    num_boxes = int(boxes.size(0))
    derived: dict[str, object] = {
        "entry_point": op,
        "num_boxes": num_boxes,
        "check": check,
    }
    if op == "boxes_iou_bev":
        derived["pairwise_overlaps"] = num_boxes * num_boxes
    else:
        derived["nms_threshold"] = NMS_THRESHOLD
    record = make_record(
        operator=op,
        case=str(fixture["case"]),
        dtype=args.dtype,
        scope=SCOPE,
        device=args.device,
        seed=args.seed,
        measurement=measurement,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        derived=derived,
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
