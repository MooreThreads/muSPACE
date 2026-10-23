"""Deterministic BEVFusion BEV IoU/NMS benchmark cases.

The three kernels that share this directory consume the same boxes but do different work, so a
published case id qualifies the fixture with the entry point it belongs to: ``n256-iou`` is the
pairwise IoU workload and ``n256-nms`` is the rotated-NMS bitmask workload over the same fixture.
A case id is stable across releases.  Every published input is built here from the documented
seed; no data set is redistributed.
"""

from __future__ import annotations

import math

import torch

FORMAT = "boxes_iou_bev_v1"

#: Fixture sweep: the number of boxes in a workload.  A fixture id is stable across releases.
DEFAULT_CASES = {
    "n32": 32,
    "n64": 64,
    "n128": 128,
    "n256": 256,
    "n512": 512,
    "n1024": 1024,
    "n2048": 2048,
}

#: Entry points that share this directory, and the suffix each one adds to its case ids.
ENTRY_POINTS = ("boxes_iou_bev", "nms_bev", "nms_bev_normal")
ENTRY_SUFFIX = {
    "boxes_iou_bev": "iou",
    "nms_bev": "nms",
    "nms_bev_normal": "nms-normal",
}
ENTRY_OF_SUFFIX = {suffix: entry for entry, suffix in ENTRY_SUFFIX.items()}

#: Entry point a bare fixture id selects when ``--op`` is omitted.
DEFAULT_OP = "nms_bev"

#: Published workload: 256 boxes on each entry point.
DEFAULT_FIXTURE = "n256"
DEFAULT_CASE = f"{DEFAULT_FIXTURE}-{ENTRY_SUFFIX[DEFAULT_OP]}"


def fixtures() -> tuple[str, ...]:
    """Return the fixture ids of the registered sweep, in declaration order."""

    return tuple(DEFAULT_CASES)


def case_ids() -> tuple[str, ...]:
    """Return every registered case id: each fixture on each entry point, in declaration order."""

    return tuple(
        f"{fixture}-{ENTRY_SUFFIX[entry]}" for entry in ENTRY_POINTS for fixture in DEFAULT_CASES
    )


def case_choices() -> tuple[str, ...]:
    """Return the accepted ``--case`` values.

    The registered case ids come first.  A bare fixture id is accepted as shorthand for
    ``--case <fixture> --op <entry point>``; the record always carries the qualified case id.
    """

    return case_ids() + fixtures()


def split_case_id(value: str) -> tuple[str, str]:
    """Split a registered case id into ``(entry point, fixture id)``."""

    for suffix, entry in ENTRY_OF_SUFFIX.items():
        marker = f"-{suffix}"
        if value.endswith(marker) and value[: -len(marker)] in DEFAULT_CASES:
            return entry, value[: -len(marker)]
    raise KeyError(
        f"unknown case {value!r}; expected a registered case ({', '.join(case_ids())}) or a "
        f"fixture id ({', '.join(fixtures())}) selected with --op"
    )


def case_parameters(case_id: str) -> dict:
    """Return the registered parameters of ``case_id``."""

    entry, fixture = split_case_id(case_id)
    return {"case": case_id, "op": entry, "fixture": fixture, "boxes": DEFAULT_CASES[fixture]}


def entry_point_of(value: str, op: str | None = None) -> tuple[str, str, int]:
    """Return ``(entry_point, fixture_id, boxes)`` for one ``--case`` / ``--op`` selection.

    A registered case id names its own entry point, and ``op`` must agree with it when given.  A
    bare fixture id takes ``op``, or the default entry point when ``--op`` is omitted.
    """

    if value in DEFAULT_CASES:
        return op or DEFAULT_OP, value, DEFAULT_CASES[value]
    entry, fixture = split_case_id(value)
    if op is not None and op != entry:
        raise SystemExit(
            f"--case {value} is registered for --op {entry}, not {op}; "
            f"registered cases: {', '.join(case_ids())}"
        )
    return entry, fixture, DEFAULT_CASES[fixture]


def make_named_case(
    case_id: str = DEFAULT_CASE, seed: int = 2333, op: str | None = None, num_boxes: int = 0
) -> dict:
    """Build the fixture registered for ``case_id`` from ``seed``.

    ``num_boxes`` overrides the registered box count for local investigation and re-derives the
    case id, so a record always names the workload it measured.  Tensors are built on the host: the
    runner moves them to the measured device without touching dtype or layout.
    """

    entry, fixture, registered = entry_point_of(case_id, op)
    boxes = int(num_boxes) if num_boxes else registered
    payload = make_case(boxes, seed)
    payload.update(
        {
            "case": f"n{boxes}-{ENTRY_SUFFIX[entry]}",
            "op": entry,
            "fixture": fixture,
            "num_boxes": boxes,
        }
    )
    return payload


def make_case(num_boxes: int = 300, seed: int = 2333) -> dict[str, object]:
    """Build the canonical fixture: boxes ``[x1, y1, x2, y2, ry]`` and one score per box.

    Box starts are uniform in ``[-40, 40)``, sizes uniform in ``[0.5, 3.5)``, yaw uniform in
    ``[-pi/2, pi/2)`` and the scores uniform in ``[0, 1)``.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    starts = torch.rand((num_boxes, 2), generator=generator) * 80.0 - 40.0
    sizes = torch.rand((num_boxes, 2), generator=generator) * 3.0 + 0.5
    boxes = torch.cat(
        [
            starts,
            starts + sizes,
            torch.rand((num_boxes, 1), generator=generator) * math.pi - math.pi / 2,
        ],
        dim=1,
    ).to(torch.float32)
    scores = torch.rand(num_boxes, generator=generator).to(torch.float32)
    return {"format": FORMAT, "seed": seed, "boxes": boxes, "scores": scores}
