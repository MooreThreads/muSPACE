"""Compare compact BEVFormer trace JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def _close(a: float, b: float, rtol: float, atol: float) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def _metric_delta(left: dict, right: dict, key: str) -> float | None:
    if key not in left or key not in right:
        return None
    return abs(float(left[key]) - float(right[key]))


def _sample_delta(left: dict, right: dict) -> tuple[str, float] | None:
    left_values = left.get("sample_values")
    right_values = right.get("sample_values")
    if not isinstance(left_values, list) or not isinstance(right_values, list):
        return None
    count = min(len(left_values), len(right_values))
    if not count:
        return None
    deltas = [abs(float(left_values[i]) - float(right_values[i])) for i in range(count)]
    return "sample_values.max_abs", max(deltas)


def _first_tensor_delta(left_outputs: list[dict], right_outputs: list[dict]) -> tuple[str, float]:
    metrics = ("sample_sum", "sample_l1", "sample_mean", "sample_std", "sample_min", "sample_max")
    best = ("", 0.0)
    for left, right in zip(left_outputs, right_outputs):
        label = left.get("path", "out")
        sample_delta = _sample_delta(left, right)
        if sample_delta is not None and sample_delta[1] > best[1]:
            best = (f"{label}.{sample_delta[0]}", sample_delta[1])
        for metric in metrics:
            delta = _metric_delta(left, right, metric)
            if delta is not None and delta > best[1]:
                best = (f"{label}.{metric}", delta)
    return best


def main() -> None:
    args = _parse_args()
    left = json.loads(args.left.read_text(encoding="utf-8"))
    right = json.loads(args.right.read_text(encoding="utf-8"))
    print(f"left torch_musa={left.get('torch_musa')} loss={left.get('loss')}")
    print(f"right torch_musa={right.get('torch_musa')} loss={right.get('loss')}")
    print()

    print("loss deltas:")
    for key in sorted(set(left["losses"]) | set(right["losses"])):
        lv = left["losses"].get(key)
        rv = right["losses"].get(key)
        if lv is None or rv is None:
            print(f"  {key}: only one side")
            continue
        print(f"  {key}: left={lv:.9g} right={rv:.9g} abs={abs(lv - rv):.9g}")
    print()

    print("state deltas:")
    for key in sorted(set(left["state"]) & set(right["state"])):
        metric, delta = _first_tensor_delta([left["state"][key]], [right["state"][key]])
        print(f"  {key}: {metric} abs={delta:.9g}")
    print()

    print("first forward records outside tolerance:")
    count = 0
    for left_record, right_record in zip(left["records"], right["records"]):
        if left_record["name"] != right_record["name"]:
            print(
                f"  order mismatch {left_record['order']}: {left_record['name']} vs {right_record['name']}"
            )
            break
        metric, delta = _first_tensor_delta(left_record["outputs"], right_record["outputs"])
        if not metric:
            continue
        left_value = None
        right_value = None
        path, metric_key = metric.rsplit(".", 1)
        for item in left_record["outputs"]:
            if item.get("path") == path:
                left_value = item.get(metric_key)
                break
        for item in right_record["outputs"]:
            if item.get("path") == path:
                right_value = item.get(metric_key)
                break
        if left_value is None or right_value is None:
            continue
        if _close(float(left_value), float(right_value), args.rtol, args.atol):
            continue
        print(
            f"  #{left_record['order']:04d} {left_record['name']}: "
            f"{metric} left={float(left_value):.9g} right={float(right_value):.9g} abs={delta:.9g}"
        )
        count += 1
        if count >= args.top:
            break


if __name__ == "__main__":
    main()
