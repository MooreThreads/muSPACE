"""Deterministic QCNet / Argoverse 2 radius neighborhood fixtures.

The three registered cases share one synthetic scene at the scale of the QCNet scene
encoder: a batch of Argoverse 2 scenarios, each contributing every historical step of its
tracked agents and its map polygons.  Positions are drawn from a fixed seed inside a
250 m x 250 m local-map box, so the 50 m query radius covers the Argoverse 2 neighborhood
scale without ever saturating the 300-neighbor cap: at most 60 agent-step points share a
segment, so a query point can never see more than 60 neighbors.

The fixture is valid for the native path: FP32, ``coordinate_dim < 9``, segment labels
that are sorted, non-negative and inside the int32 indexing range, and point counts below
``2**31 - 1``.
"""

from __future__ import annotations

import torch

FORMAT = "muspace.radius.v1"

#: Modelled Argoverse 2 scene: one scenario per benchmark batch, 50 historical steps at
#: 10 Hz and 60 tracked agents, with the scenario map summarized as 12000 polygon
#: vertices (lane-segment left boundaries, right boundaries and centerlines, plus
#: crosswalks).  QCNet repeats the map over the historical steps before it queries it,
#: which is what makes the polygon query set large.
SCENARIOS = 1
HISTORICAL_STEPS = 50
AGENTS_PER_SCENARIO = 60
POLYGONS_PER_SCENARIO = 12_000
COORDINATE_DIM = 2
LOCAL_MAP_EXTENT_M = 250.0

#: The QCNet scene encoder queries both neighborhood families with a 50 m radius and a
#: 300-neighbor cap.
RADIUS_M = 50.0
MAX_NUM_NEIGHBORS = 300

#: Case ids are stable across releases: an id always describes the same workload and the
#: same entry point.  The three cases share the scene above and differ in the
#: neighborhood that the scene encoder builds from it.
CASES: dict[str, dict] = {
    "av2-enc-pl2a": {
        "entry_point": "radius",
        "description": (
            "QCNet scene encoder polygon-to-agent query: every polygon-step point asks "
            "which agent-step points of its own historical step lie within 50 m"
        ),
    },
    "av2-enc-a2a": {
        "entry_point": "radius_graph",
        "description": (
            "QCNet scene encoder agent-to-agent graph: every agent-step point is linked to "
            "the other agent-step points of its own historical step within 50 m"
        ),
    },
    "av2-enc-compat": {
        "entry_point": "qcnet_compat",
        "description": (
            "QCNet scene encoder neighborhood construction through mu_space.qcnet_compat: "
            "the polygon-to-agent query and the agent-to-agent graph of one encoder step, "
            "issued with the QCNet call signature"
        ),
    },
}
DEFAULT_CASE = "av2-enc-pl2a"


def case_ids() -> tuple[str, ...]:
    """Return the canonical case ids in declaration order."""

    return tuple(CASES)


def case_parameters(case_id: str = DEFAULT_CASE) -> dict:
    """Return the registered parameters of ``case_id``."""

    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown radius case {case_id!r}; expected one of {', '.join(case_ids())}"
        ) from None


def segment_labels(segments: int, points_per_segment: int, points: int) -> torch.Tensor:
    """Return the QCNet segment labels of one encoder point set.

    QCNet labels every encoder point with ``scenario + scenarios * step``, so the labels
    of one batch are a repeat-interleave of the segment index.  The helper keeps the
    fixture inside the sorted, non-negative, int32 indexing range that the native path
    documents, and uses the int64 dtype that PyG batching produces for QCNet.
    """

    if points >= 2**31 - 1:
        raise ValueError("radius fixtures must stay inside the int32 indexing range")
    labels = torch.arange(segments, dtype=torch.int64).repeat_interleave(points_per_segment)
    if labels.numel() != points:
        raise ValueError(f"expected {points} segment labels, built {labels.numel()}")
    if not bool((labels >= 0).all()) or not bool((labels[1:] >= labels[:-1]).all()):
        raise ValueError("radius fixtures require sorted and non-negative segment labels")
    return labels


def _scene(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the agent and polygon positions of every scenario and historical step."""

    generator = torch.Generator().manual_seed(seed)
    half_extent = LOCAL_MAP_EXTENT_M / 2.0

    def draw(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            torch.rand(shape, generator=generator, dtype=torch.float32) * LOCAL_MAP_EXTENT_M
            - half_extent
        )

    agents = draw((SCENARIOS, HISTORICAL_STEPS, AGENTS_PER_SCENARIO, COORDINATE_DIM))
    polygons = draw((SCENARIOS, POLYGONS_PER_SCENARIO, COORDINATE_DIM))
    return agents, polygons


def make_named_case(case_id: str = DEFAULT_CASE, seed: int = 2333) -> dict:
    """Build the scene registered for ``case_id`` from ``seed``."""

    parameters = case_parameters(case_id)
    agents, polygons = _scene(seed)
    segments = SCENARIOS * HISTORICAL_STEPS
    # QCNet orders both point sets step-major, and the scenario index varies fastest
    # inside a step, so the label of encoder point ``(step, scenario)`` is
    # ``scenario + scenarios * step``.
    x = agents.permute(1, 0, 2, 3).reshape(-1, COORDINATE_DIM).contiguous()
    y = polygons.reshape(-1, COORDINATE_DIM).repeat(HISTORICAL_STEPS, 1).contiguous()
    return {
        "format": FORMAT,
        "case": case_id,
        "seed": seed,
        "entry_point": parameters["entry_point"],
        "description": parameters["description"],
        "scalars": {
            "scenarios": SCENARIOS,
            "historical_steps": HISTORICAL_STEPS,
            "agents_per_scenario": AGENTS_PER_SCENARIO,
            "polygons_per_scenario": POLYGONS_PER_SCENARIO,
            "coordinate_dim": COORDINATE_DIM,
            "local_map_extent_m": LOCAL_MAP_EXTENT_M,
            "radius": RADIUS_M,
            "max_num_neighbors": MAX_NUM_NEIGHBORS,
        },
        "inputs": {
            "x": x,
            "y": y,
            "batch_x": segment_labels(segments, AGENTS_PER_SCENARIO, x.shape[0]),
            "batch_y": segment_labels(segments, POLYGONS_PER_SCENARIO, y.shape[0]),
        },
    }


def make_case(seed: int = 2333) -> dict:
    """Build the canonical radius case; the public helper of this module."""

    return make_named_case(DEFAULT_CASE, seed=seed)
