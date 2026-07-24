#!/usr/bin/env python3
"""Diagnose MobileManiBench dual-plan predictions without retraining.

The script consumes the physical-unit arrays written by
``evaluate_mobilemanibench_plan.py`` and the conversion-time ``plan_stats.json``.
It checks normalization round trips, reports slice-wise errors/saturation, ranks
samples, and renders representative high/low-error Base/EEF trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--plan-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=5)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def masked_values(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    mask = np.asarray(valid, dtype=bool)
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    return values[np.broadcast_to(mask, values.shape)]


def q_bounds(stats: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    item = stats["statistics"][name]
    return np.asarray(item["q01"], dtype=np.float64), np.asarray(
        item["q99"], dtype=np.float64
    )


def q_normalize_raw(
    values: np.ndarray, q01: np.ndarray, q99: np.ndarray
) -> np.ndarray:
    width = q99 - q01
    safe_width = np.where(width != 0.0, width, 1.0)
    normalized = 2.0 * (values - q01) / safe_width - 1.0
    return np.where(width != 0.0, normalized, 0.0)


def q_normalize_train(
    values: np.ndarray, q01: np.ndarray, q99: np.ndarray
) -> np.ndarray:
    return np.clip(q_normalize_raw(values, q01, q99), -1.0, 1.0)


def q_inverse(
    values: np.ndarray, q01: np.ndarray, q99: np.ndarray
) -> np.ndarray:
    width = q99 - q01
    reconstructed = (values + 1.0) * 0.5 * width + q01
    return np.where(width != 0.0, reconstructed, q01)


def roundtrip_slice(
    values: np.ndarray,
    valid: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    raw = q_normalize_raw(values, q01, q99)
    normalized = np.clip(raw, -1.0, 1.0)
    reconstructed = q_inverse(normalized, q01, q99)
    error = np.abs(reconstructed - values)
    inside = (values >= q01) & (values <= q99)
    result: dict[str, Any] = {
        "all_dimensions": finite_summary(masked_values(error, valid)),
        "clipped_fraction": float(
            np.mean(masked_values(~inside, valid).astype(np.float64))
        ),
        "per_dimension": {},
    }
    for index, name in enumerate(names):
        active = np.asarray(valid, dtype=bool)
        dim_error = error[..., index][active]
        dim_inside = inside[..., index][active]
        result["per_dimension"][name] = {
            "all": finite_summary(dim_error),
            "inside_q01_q99": finite_summary(dim_error[dim_inside]),
            "clipped_fraction": float(np.mean(~dim_inside)),
        }
    return result


def normalized_slice_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    # Prediction was inverse-mapped without clipping by MobilePlanTransform.
    # Reconstructing raw normalized values recovers the sampler output.
    pred_normalized = q_normalize_raw(prediction, q01, q99)
    target_raw = q_normalize_raw(target, q01, q99)
    target_normalized = np.clip(target_raw, -1.0, 1.0)
    error = pred_normalized - target_normalized
    result: dict[str, Any] = {
        "normalized_mae": finite_summary(masked_values(np.abs(error), valid)),
        "normalized_mse": finite_summary(masked_values(np.square(error), valid)),
        "target_saturation_abs_ge_0_95": float(
            np.mean(
                masked_values(np.abs(target_normalized) >= 0.95, valid).astype(
                    np.float64
                )
            )
        ),
        "target_outside_q01_q99": float(
            np.mean(masked_values(np.abs(target_raw) > 1.0, valid).astype(np.float64))
        ),
        "prediction_saturation_abs_ge_0_95": float(
            np.mean(
                masked_values(np.abs(pred_normalized) >= 0.95, valid).astype(
                    np.float64
                )
            )
        ),
        "prediction_outside_normalized_range": float(
            np.mean(
                masked_values(np.abs(pred_normalized) > 1.0, valid).astype(
                    np.float64
                )
            )
        ),
        "per_dimension": {},
    }
    for index, name in enumerate(names):
        active = np.asarray(valid, dtype=bool)
        physical_error = prediction[..., index] - target[..., index]
        normalized_error = error[..., index]
        result["per_dimension"][name] = {
            "physical_bias": float(np.mean(physical_error[active])),
            "physical_mae": finite_summary(np.abs(physical_error[active])),
            "physical_rmse": float(
                np.sqrt(np.mean(np.square(physical_error[active])))
            ),
            "normalized_mae": finite_summary(np.abs(normalized_error[active])),
            "normalized_mse": finite_summary(np.square(normalized_error[active])),
            "target_saturation_abs_ge_0_95": float(
                np.mean(np.abs(target_normalized[..., index][active]) >= 0.95)
            ),
            "target_outside_q01_q99": float(
                np.mean(np.abs(target_raw[..., index][active]) > 1.0)
            ),
            "prediction_saturation_abs_ge_0_95": float(
                np.mean(np.abs(pred_normalized[..., index][active]) >= 0.95)
            ),
            "prediction_outside_normalized_range": float(
                np.mean(np.abs(pred_normalized[..., index][active]) > 1.0)
            ),
        }
    return result


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    active = np.asarray(valid, dtype=bool)
    for index, name in enumerate(names):
        pred = prediction[..., index][active].astype(np.float64)
        gt = target[..., index][active].astype(np.float64)
        gt_variance = float(np.var(gt))
        correlation = (
            float(np.corrcoef(gt, pred)[0, 1]) if gt_variance > 0.0 else 0.0
        )
        slope = (
            float(np.cov(gt, pred, bias=True)[0, 1] / gt_variance)
            if gt_variance > 0.0
            else 0.0
        )
        intercept = float(np.mean(pred) - slope * np.mean(gt))
        result[name] = {
            "gt_mean": float(np.mean(gt)),
            "prediction_mean": float(np.mean(pred)),
            "bias": float(np.mean(pred - gt)),
            "gt_std": float(np.std(gt)),
            "prediction_std": float(np.std(pred)),
            "prediction_to_gt_std_ratio": (
                float(np.std(pred) / np.std(gt)) if np.std(gt) > 0.0 else 0.0
            ),
            "pearson_correlation": correlation,
            "linear_slope": slope,
            "linear_intercept": intercept,
        }
    return result


def trajectory_length_metrics(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> dict[str, Any]:
    prediction_lengths: list[float] = []
    target_lengths: list[float] = []
    for index in range(len(valid)):
        active = valid[index]
        if np.sum(active) < 2:
            continue
        pred_points = prediction[index, active]
        target_points = target[index, active]
        prediction_lengths.append(
            float(np.linalg.norm(np.diff(pred_points, axis=0), axis=-1).sum())
        )
        target_lengths.append(
            float(np.linalg.norm(np.diff(target_points, axis=0), axis=-1).sum())
        )
    prediction_array = np.asarray(prediction_lengths)
    target_array = np.asarray(target_lengths)
    return {
        "prediction_m": finite_summary(prediction_array),
        "target_m": finite_summary(target_array),
        "mean_prediction_to_target_ratio": float(
            np.mean(prediction_array) / np.mean(target_array)
        ),
    }


def normalize_rows(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    fallback = np.zeros_like(vector)
    fallback[..., 0] = 1.0
    return np.where(norm > eps, vector / np.maximum(norm, eps), fallback)


def rotation6d_rows_to_matrix(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64).reshape(*value.shape[:-1], 2, 3)
    first = normalize_rows(rows[..., 0, :])
    second_raw = rows[..., 1, :] - np.sum(
        rows[..., 1, :] * first, axis=-1, keepdims=True
    ) * first
    second = normalize_rows(second_raw)
    third = normalize_rows(np.cross(first, second))
    second = normalize_rows(np.cross(third, first))
    return np.stack([first, second, third], axis=-2)


def rotation_geodesic_deg(
    prediction: np.ndarray, target: np.ndarray
) -> np.ndarray:
    relative = prediction @ np.swapaxes(target, -1, -2)
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cosine))


def rotation6d_quality(values: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    rows = np.asarray(values, dtype=np.float64).reshape(*values.shape[:-1], 2, 3)
    first_norm = np.linalg.norm(rows[..., 0, :], axis=-1)
    second_norm = np.linalg.norm(rows[..., 1, :], axis=-1)
    row_dot = np.sum(rows[..., 0, :] * rows[..., 1, :], axis=-1)
    cross_norm = np.linalg.norm(
        np.cross(rows[..., 0, :], rows[..., 1, :]), axis=-1
    )
    active = np.asarray(valid, dtype=bool)
    return {
        "first_row_norm": finite_summary(first_norm[active]),
        "second_row_norm": finite_summary(second_norm[active]),
        "absolute_row_dot": finite_summary(np.abs(row_dot[active])),
        "cross_norm": finite_summary(cross_norm[active]),
        "near_degenerate_cross_lt_0_1_fraction": float(
            np.mean(cross_norm[active] < 0.1)
        ),
    }


def per_horizon_mean(values: np.ndarray, valid: np.ndarray) -> list[float]:
    output: list[float] = []
    for horizon in range(valid.shape[1]):
        active = valid[:, horizon]
        output.append(float(np.mean(values[:, horizon][active])))
    return output


def sample_metrics(
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    base_pred: np.ndarray,
    base_gt: np.ndarray,
    manip_pred: np.ndarray,
    manip_gt: np.ndarray,
    valid: np.ndarray,
    hand_dim: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    base_position = np.linalg.norm(base_pred[..., :2] - base_gt[..., :2], axis=-1)
    eef_position = np.linalg.norm(manip_pred[..., :3] - manip_gt[..., :3], axis=-1)
    pred_rotation = rotation6d_rows_to_matrix(manip_pred[..., 3:9])
    gt_rotation = rotation6d_rows_to_matrix(manip_gt[..., 3:9])
    orientation = rotation_geodesic_deg(pred_rotation, gt_rotation)
    if hand_dim:
        hand = np.mean(
            np.abs(
                manip_pred[..., 9 : 9 + hand_dim]
                - manip_gt[..., 9 : 9 + hand_dim]
            ),
            axis=-1,
        )
    else:
        hand = np.zeros_like(eef_position)

    valid_eef = eef_position[valid]
    valid_orientation = orientation[valid]
    valid_hand = hand[valid]
    scales = {
        "eef_position": max(float(np.median(valid_eef)), 1e-8),
        "orientation": max(float(np.median(valid_orientation)), 1e-8),
        "hand": max(float(np.median(valid_hand)), 1e-8),
    }

    rows: list[dict[str, Any]] = []
    for index in range(len(episode_index)):
        active = valid[index]
        if not np.any(active):
            continue
        eef_mean = float(np.mean(eef_position[index, active]))
        orientation_mean = float(np.mean(orientation[index, active]))
        hand_mean = float(np.mean(hand[index, active]))
        score = (
            eef_mean / scales["eef_position"]
            + orientation_mean / scales["orientation"]
            + hand_mean / scales["hand"]
        ) / 3.0
        rows.append(
            {
                "array_index": index,
                "episode_index": int(episode_index[index]),
                "frame_index": int(frame_index[index]),
                "valid_waypoints": int(np.sum(active)),
                "composite_score": score,
                "base_position_mean_error_m": float(
                    np.mean(base_position[index, active])
                ),
                "eef_position_mean_error_m": eef_mean,
                "eef_orientation_mean_error_deg": orientation_mean,
                "hand_joint_mae": hand_mean,
            }
        )
    rows.sort(key=lambda row: row["composite_score"])
    return rows, {
        "base_position": base_position,
        "eef_position": eef_position,
        "eef_orientation": orientation,
        "hand": hand,
    }


def set_equal_3d_axes(axis: Any, points: np.ndarray) -> None:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 0.05)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def plot_sample(
    output_path: Path,
    row: dict[str, Any],
    offsets: np.ndarray,
    base_pred: np.ndarray,
    base_gt: np.ndarray,
    manip_pred: np.ndarray,
    manip_gt: np.ndarray,
    valid: np.ndarray,
    errors: dict[str, np.ndarray],
) -> None:
    index = int(row["array_index"])
    active = valid[index]
    active_offsets = offsets[active]
    bp = base_pred[index, active]
    bg = base_gt[index, active]
    mp = manip_pred[index, active]
    mg = manip_gt[index, active]
    rp = rotation6d_rows_to_matrix(mp[:, 3:9])
    rg = rotation6d_rows_to_matrix(mg[:, 3:9])

    figure = plt.figure(figsize=(14, 10), constrained_layout=True)
    figure.suptitle(
        "episode={episode_index} frame={frame_index} "
        "composite={composite_score:.3f}".format(**row)
    )

    base_axis = figure.add_subplot(2, 2, 1)
    base_axis.plot(bg[:, 0], bg[:, 1], "o-", color="black", label="Base GT")
    base_axis.plot(bp[:, 0], bp[:, 1], "o--", color="tab:orange", label="Base pred")
    for values, color in ((bg, "black"), (bp, "tab:orange")):
        yaw = np.arctan2(values[:, 2], values[:, 3])
        base_axis.quiver(
            values[:, 0],
            values[:, 1],
            np.cos(yaw),
            np.sin(yaw),
            angles="xy",
            scale_units="xy",
            scale=20.0,
            color=color,
            alpha=0.7,
        )
    for offset, x_value, y_value in zip(active_offsets, bg[:, 0], bg[:, 1]):
        base_axis.annotate(str(int(offset)), (x_value, y_value))
    base_axis.set_title("Base trajectory in anchor B(t)")
    base_axis.set_xlabel("x [m]")
    base_axis.set_ylabel("y [m]")
    base_axis.axis("equal")
    base_axis.grid(True, alpha=0.3)
    base_axis.legend()

    eef_axis = figure.add_subplot(2, 2, 2, projection="3d")
    eef_axis.plot(
        mg[:, 0], mg[:, 1], mg[:, 2], "o-", color="black", label="EEF GT"
    )
    eef_axis.plot(
        mp[:, 0],
        mp[:, 1],
        mp[:, 2],
        "o--",
        color="tab:orange",
        label="EEF pred",
    )
    axis_colors = ("tab:red", "tab:green", "tab:blue")
    axis_scale = 0.035
    for point, rotation in zip(mg[:, :3], rg):
        for axis_index, color in enumerate(axis_colors):
            direction = rotation[axis_index]
            eef_axis.plot(
                [point[0], point[0] + axis_scale * direction[0]],
                [point[1], point[1] + axis_scale * direction[1]],
                [point[2], point[2] + axis_scale * direction[2]],
                color=color,
                linewidth=1.5,
            )
    for point, rotation in zip(mp[:, :3], rp):
        for axis_index, color in enumerate(axis_colors):
            direction = rotation[axis_index]
            eef_axis.plot(
                [point[0], point[0] + axis_scale * direction[0]],
                [point[1], point[1] + axis_scale * direction[1]],
                [point[2], point[2] + axis_scale * direction[2]],
                color=color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
            )
    set_equal_3d_axes(eef_axis, np.concatenate([mg[:, :3], mp[:, :3]], axis=0))
    eef_axis.set_title("EEF trajectory/orientation (solid GT, dashed pred)")
    eef_axis.set_xlabel("x [m]")
    eef_axis.set_ylabel("y [m]")
    eef_axis.set_zlabel("z [m]")
    eef_axis.legend()

    error_axis = figure.add_subplot(2, 2, 3)
    error_axis.plot(
        active_offsets,
        100.0 * errors["base_position"][index, active],
        "o-",
        label="Base position [cm]",
    )
    error_axis.plot(
        active_offsets,
        100.0 * errors["eef_position"][index, active],
        "o-",
        label="EEF position [cm]",
    )
    error_axis.plot(
        active_offsets,
        errors["eef_orientation"][index, active],
        "o-",
        label="EEF orientation [deg]",
    )
    error_axis.set_title("Errors by plan offset")
    error_axis.set_xlabel("offset [control steps]")
    error_axis.grid(True, alpha=0.3)
    error_axis.legend()

    slice_axis = figure.add_subplot(2, 2, 4)
    xyz_abs = np.abs(mp[:, :3] - mg[:, :3])
    labels = ["x", "y", "z"]
    for dim, label in enumerate(labels):
        slice_axis.plot(
            active_offsets,
            100.0 * xyz_abs[:, dim],
            "o-",
            label=f"EEF {label} abs error [cm]",
        )
    if mp.shape[-1] > 9:
        slice_axis.plot(
            active_offsets,
            np.abs(mp[:, 9] - mg[:, 9]),
            "o-",
            label="gripper abs error",
        )
    slice_axis.set_title("Manipulator physical slice errors")
    slice_axis.set_xlabel("offset [control steps]")
    slice_axis.grid(True, alpha=0.3)
    slice_axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_aggregate(
    output_path: Path,
    offsets: np.ndarray,
    valid: np.ndarray,
    errors: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = [
        ("base_position", 100.0, "Base position error [cm]"),
        ("eef_position", 100.0, "EEF position error [cm]"),
        ("eef_orientation", 1.0, "EEF orientation error [deg]"),
        ("hand", 1.0, "Hand joint MAE"),
    ]
    for axis, (name, scale, title) in zip(axes.flat, panels):
        means = []
        medians = []
        p90s = []
        for horizon in range(valid.shape[1]):
            values = errors[name][:, horizon][valid[:, horizon]] * scale
            means.append(np.mean(values))
            medians.append(np.median(values))
            p90s.append(np.quantile(values, 0.9))
        axis.plot(offsets, means, "o-", label="mean")
        axis.plot(offsets, medians, "o-", label="median")
        axis.plot(offsets, p90s, "o-", label="p90")
        axis.set_title(title)
        axis.set_xlabel("offset [control steps]")
        axis.grid(True, alpha=0.3)
        axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_ranking(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: float) -> str:
    return f"{value:.6g}"


def write_markdown_report(
    path: Path,
    report: dict[str, Any],
    low_rows: list[dict[str, Any]],
    high_rows: list[dict[str, Any]],
) -> None:
    xyz = report["slice_metrics"]["eef_xyz"]["per_dimension"]
    hand = report["slice_metrics"]["hand"]["per_dimension"]
    rotation = report["slice_metrics"]["rotation6d"]
    roundtrip = report["roundtrip"]
    lines = [
        "# MobileManiBench Plan Prediction Diagnostics",
        "",
        "## Normalization round trip",
        "",
        "| Slice | all-value MAE | in-range max error | clipped fraction |",
        "|---|---:|---:|---:|",
    ]
    for name in ("base_xy", "eef_xyz", "hand"):
        item = roundtrip[name]
        inside_max = max(
            dim["inside_q01_q99"]["max"]
            for dim in item["per_dimension"].values()
        )
        lines.append(
            f"| {name} | {format_metric(item['all_dimensions']['mean'])} | "
            f"{format_metric(inside_max)} | "
            f"{100.0 * item['clipped_fraction']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "Identity slices (`base_yaw_sincos`, `eef_rotation6d`) have exact "
            "round-trip by construction.",
            "",
            "## Manipulator slice metrics",
            "",
            "| Slice | physical MAE | normalized MAE | GT saturation >=0.95 | "
            "prediction outside [-1,1] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in xyz.items():
        lines.append(
            f"| EEF {name} | {format_metric(item['physical_mae']['mean'])} m | "
            f"{format_metric(item['normalized_mae']['mean'])} | "
            f"{100.0 * item['target_saturation_abs_ge_0_95']:.3f}% | "
            f"{100.0 * item['prediction_outside_normalized_range']:.3f}% |"
        )
    for name, item in hand.items():
        lines.append(
            f"| Hand {name} | {format_metric(item['physical_mae']['mean'])} | "
            f"{format_metric(item['normalized_mae']['mean'])} | "
            f"{100.0 * item['target_saturation_abs_ge_0_95']:.3f}% | "
            f"{100.0 * item['prediction_outside_normalized_range']:.3f}% |"
        )
    lines.extend(
        [
            f"| rotation6d (6D mean) | "
            f"{format_metric(rotation['raw_6d_mae']['mean'])} | "
            f"{format_metric(rotation['raw_6d_mae']['mean'])} | identity | identity |",
            "",
            f"Rotation geodesic mean: "
            f"{format_metric(rotation['geodesic_deg']['mean'])} deg.",
            "",
            "## Prediction/GT scale and correlation",
            "",
            "| Slice | bias | pred/GT std | correlation | linear slope |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    regression = report["distribution_diagnostics"]["regression"]
    for name, item in regression.items():
        lines.append(
            f"| {name} | {item['bias']:.6g} | "
            f"{item['prediction_to_gt_std_ratio']:.4f} | "
            f"{item['pearson_correlation']:.4f} | "
            f"{item['linear_slope']:.4f} |"
        )
    path_length = report["distribution_diagnostics"]["eef_trajectory_length"]
    lines.extend(
        [
            "",
            "EEF trajectory length over the six waypoints:",
            "",
            f"- GT mean: {path_length['target_m']['mean']:.6g} m",
            f"- prediction mean: {path_length['prediction_m']['mean']:.6g} m",
            f"- prediction/GT ratio: "
            f"{path_length['mean_prediction_to_target_ratio']:.4f}",
            "",
            "## Representative samples",
            "",
            "### Lowest composite errors",
            "",
            "| rank | episode | frame | score | EEF position (m) | "
            "EEF orientation (deg) | hand MAE |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(low_rows, start=1):
        lines.append(
            f"| {rank} | {row['episode_index']} | {row['frame_index']} | "
            f"{row['composite_score']:.4f} | "
            f"{row['eef_position_mean_error_m']:.4f} | "
            f"{row['eef_orientation_mean_error_deg']:.2f} | "
            f"{row['hand_joint_mae']:.4f} |"
        )
    lines.extend(
        [
            "",
            "### Highest composite errors",
            "",
            "| rank | episode | frame | score | EEF position (m) | "
            "EEF orientation (deg) | hand MAE |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(high_rows, start=1):
        lines.append(
            f"| {rank} | {row['episode_index']} | {row['frame_index']} | "
            f"{row['composite_score']:.4f} | "
            f"{row['eef_position_mean_error_m']:.4f} | "
            f"{row['eef_orientation_mean_error_deg']:.2f} | "
            f"{row['hand_joint_mae']:.4f} |"
        )
    lines.extend(
        [
            "",
            "See `figures/` for Base/EEF trajectories and orientation axes. "
            "The composite score is used only for ranking and is the mean of "
            "median-normalized EEF position, orientation, and hand errors.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = np.load(args.predictions)
    stats = json.loads(args.plan_stats.read_text(encoding="utf-8"))

    episode_index = predictions["episode_index"]
    frame_index = predictions["frame_index"]
    base_pred = predictions["base_pred"].astype(np.float64)
    base_gt = predictions["base_gt"].astype(np.float64)
    manip_pred = predictions["manipulator_pred"].astype(np.float64)
    manip_gt = predictions["manipulator_gt"].astype(np.float64)
    valid = predictions["plan_valid"].astype(bool)
    offsets = np.asarray(stats["plan_time_offsets"], dtype=np.int64)
    hand_dim = int(stats["hand_dim"])
    manipulator_dim = int(stats["manipulator_dim"])

    base_q01, base_q99 = q_bounds(stats, "base_xy")
    eef_q01, eef_q99 = q_bounds(stats, "eef_xyz")
    hand_q01, hand_q99 = q_bounds(stats, "hand")

    roundtrip = {
        "base_xy": roundtrip_slice(
            base_gt[..., :2],
            valid,
            base_q01,
            base_q99,
            ["x", "y"],
        ),
        "eef_xyz": roundtrip_slice(
            manip_gt[..., :3],
            valid,
            eef_q01,
            eef_q99,
            ["x", "y", "z"],
        ),
        "hand": roundtrip_slice(
            manip_gt[..., 9 : 9 + hand_dim],
            valid,
            hand_q01,
            hand_q99,
            [f"joint_{index}" for index in range(hand_dim)],
        ),
        "base_yaw_sincos": {
            "policy": "identity",
            "max_error": 0.0,
        },
        "eef_rotation6d": {
            "policy": "identity",
            "max_error": 0.0,
        },
    }

    eef_metrics = normalized_slice_metrics(
        manip_pred[..., :3],
        manip_gt[..., :3],
        valid,
        eef_q01,
        eef_q99,
        ["x", "y", "z"],
    )
    hand_metrics = normalized_slice_metrics(
        manip_pred[..., 9 : 9 + hand_dim],
        manip_gt[..., 9 : 9 + hand_dim],
        valid,
        hand_q01,
        hand_q99,
        [f"joint_{index}" for index in range(hand_dim)],
    )
    rotation_difference = manip_pred[..., 3:9] - manip_gt[..., 3:9]
    rotation_geodesic = rotation_geodesic_deg(
        rotation6d_rows_to_matrix(manip_pred[..., 3:9]),
        rotation6d_rows_to_matrix(manip_gt[..., 3:9]),
    )
    rotation_metrics = {
        "raw_6d_mae": finite_summary(
            masked_values(np.abs(rotation_difference), valid)
        ),
        "raw_6d_mse": finite_summary(
            masked_values(np.square(rotation_difference), valid)
        ),
        "geodesic_deg": finite_summary(rotation_geodesic[valid]),
        "prediction_geometry": rotation6d_quality(
            manip_pred[..., 3:9], valid
        ),
        "target_geometry": rotation6d_quality(manip_gt[..., 3:9], valid),
    }

    rankings, errors = sample_metrics(
        episode_index,
        frame_index,
        base_pred,
        base_gt,
        manip_pred,
        manip_gt,
        valid,
        hand_dim,
    )
    # Compare representative samples on equal footing. Episode-tail anchors
    # with fewer valid horizons remain in sample_ranking.csv but are excluded
    # from the high/low visualization sets.
    full_horizon_rankings = [
        row
        for row in rankings
        if row["valid_waypoints"] == int(len(offsets))
    ]
    num_examples = min(args.num_examples, len(full_horizon_rankings))
    low_rows = full_horizon_rankings[:num_examples]
    high_rows = list(reversed(full_horizon_rankings[-num_examples:]))

    report = {
        "inputs": {
            "predictions": str(args.predictions),
            "plan_stats": str(args.plan_stats),
            "num_samples": int(len(episode_index)),
            "num_valid_waypoints": int(np.sum(valid)),
            "manipulator_dim": manipulator_dim,
            "hand_dim": hand_dim,
            "plan_offsets": offsets.tolist(),
        },
        "roundtrip": roundtrip,
        "slice_metrics": {
            "eef_xyz": eef_metrics,
            "rotation6d": rotation_metrics,
            "hand": hand_metrics,
        },
        "distribution_diagnostics": {
            "regression": {
                **regression_metrics(
                    manip_pred[..., :3],
                    manip_gt[..., :3],
                    valid,
                    ["eef_x", "eef_y", "eef_z"],
                ),
                **regression_metrics(
                    manip_pred[..., 9 : 9 + hand_dim],
                    manip_gt[..., 9 : 9 + hand_dim],
                    valid,
                    [f"hand_joint_{index}" for index in range(hand_dim)],
                ),
            },
            "eef_trajectory_length": trajectory_length_metrics(
                manip_pred[..., :3], manip_gt[..., :3], valid
            ),
        },
        "per_horizon": {
            "offsets": offsets.tolist(),
            "eef_position_mean_m": per_horizon_mean(
                errors["eef_position"], valid
            ),
            "eef_orientation_mean_deg": per_horizon_mean(
                errors["eef_orientation"], valid
            ),
            "hand_mae": per_horizon_mean(errors["hand"], valid),
        },
        "lowest_error_samples": low_rows,
        "highest_error_samples": high_rows,
    }
    write_json(args.output_dir / "diagnostic_summary.json", report)
    write_ranking(args.output_dir / "sample_ranking.csv", rankings)
    write_markdown_report(
        args.output_dir / "diagnostic_report.md",
        report,
        low_rows,
        high_rows,
    )
    plot_aggregate(
        args.output_dir / "figures" / "aggregate_horizon_errors.png",
        offsets,
        valid,
        errors,
    )
    for label, rows in (("low", low_rows), ("high", high_rows)):
        for rank, row in enumerate(rows, start=1):
            filename = (
                f"{label}_{rank:02d}_ep{row['episode_index']}_"
                f"frame{row['frame_index']}.png"
            )
            plot_sample(
                args.output_dir / "figures" / filename,
                row,
                offsets,
                base_pred,
                base_gt,
                manip_pred,
                manip_gt,
                valid,
                errors,
            )
    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
