#!/usr/bin/env python3
"""Aggregate physical-loss gradient probes into reproducible Hydra weights."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


PLAN_COMPONENT_KEY = "recommended_plan_component_loss_weight_metric_avg"
BASE_EEF_CONSISTENCY_KEY = (
    "recommended_base_eef_consistency_loss_weight_metric_avg"
)
PRIOR_KEY = "recommended_base_prior_loss_weight_metric_avg"
EEF_PRIOR_KEY = "recommended_eef_prior_loss_weight_metric_avg"
JOINT_PRIOR_KEY = (
    "recommended_joint_prior_consistency_loss_weight_metric_avg"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trainer_states",
        nargs="+",
        type=Path,
        help="trainer_state.json files from short gradient-probe runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--weight-min", type=float, default=1e-4)
    parser.add_argument("--weight-max", type=float, default=10.0)
    return parser.parse_args()


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value) and value > 0]


def collect(paths: list[Path]) -> dict[str, list[float]]:
    result = {
        PLAN_COMPONENT_KEY: [],
        BASE_EEF_CONSISTENCY_KEY: [],
        PRIOR_KEY: [],
        EEF_PRIOR_KEY: [],
        JOINT_PRIOR_KEY: [],
    }
    for path in paths:
        state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        for entry in state.get("log_history", []):
            for key in result:
                if key in entry:
                    result[key].append(float(entry[key]))
    return {key: finite(values) for key, values in result.items()}


def robust_weight(values: list[float], minimum: float, maximum: float) -> float:
    return min(max(statistics.median(values), minimum), maximum)


def main() -> int:
    args = parse_args()
    values = collect(args.trainer_states)
    required_keys = (PLAN_COMPONENT_KEY, BASE_EEF_CONSISTENCY_KEY)
    for key in required_keys:
        samples = values[key]
        if len(samples) < args.minimum_samples:
            raise ValueError(
                f"{key}: need at least {args.minimum_samples} probes, "
                f"got {len(samples)}"
            )
    result = {
        "method": "median_output_gradient_norm_ratio",
        "trainer_states": [str(path.resolve()) for path in args.trainer_states],
        "num_plan_component_probes": len(values[PLAN_COMPONENT_KEY]),
        "num_base_eef_consistency_probes": len(
            values[BASE_EEF_CONSISTENCY_KEY]
        ),
        "plan_component_loss_weight": robust_weight(
            values[PLAN_COMPONENT_KEY], args.weight_min, args.weight_max
        ),
        "base_eef_consistency_loss_weight": robust_weight(
            values[BASE_EEF_CONSISTENCY_KEY],
            args.weight_min,
            args.weight_max,
        ),
    }
    if values[PRIOR_KEY]:
        if len(values[PRIOR_KEY]) < args.minimum_samples:
            raise ValueError(
                f"{PRIOR_KEY}: need at least {args.minimum_samples} probes, "
                f"got {len(values[PRIOR_KEY])}"
            )
        result["num_base_prior_probes"] = len(values[PRIOR_KEY])
        result["base_prior_loss_weight"] = robust_weight(
            values[PRIOR_KEY], args.weight_min, args.weight_max
        )
    optional_prior_weights = (
        (
            EEF_PRIOR_KEY,
            "num_eef_prior_probes",
            "eef_prior_loss_weight",
        ),
        (
            JOINT_PRIOR_KEY,
            "num_joint_prior_consistency_probes",
            "joint_prior_consistency_loss_weight",
        ),
    )
    for metric_key, count_key, weight_key in optional_prior_weights:
        if not values[metric_key]:
            continue
        if len(values[metric_key]) < args.minimum_samples:
            raise ValueError(
                f"{metric_key}: need at least {args.minimum_samples} probes, "
                f"got {len(values[metric_key])}"
            )
        result[count_key] = len(values[metric_key])
        result[weight_key] = robust_weight(
            values[metric_key], args.weight_min, args.weight_max
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
