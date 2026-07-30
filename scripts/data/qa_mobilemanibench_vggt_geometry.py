#!/usr/bin/env python3
"""Audit MobileManiBench pseudo-3D geometry and video/control timing."""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from groot.vla.data.dataset.mobilemanibench_vggt import (
    ISAAC_X_FORWARD_FROM_OPENCV,
    MobileManiBenchVGGTDataset,
)
from groot.vla.model.vggt_3d_wam.geometry import (
    invert_transform,
    points_in_metric_grid,
    range_to_pointmap,
    scale_intrinsics,
)


VIDEO_OFFSETS = tuple(range(33))
VIEW_NAMES = ("head", "wrist")
OVERLAP_THRESHOLDS_M = (0.05, 0.10, 0.15, 0.25, 0.50)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--optical-search-samples", type=int, default=20)
    parser.add_argument("--ffprobe-episodes", type=int, default=10)
    parser.add_argument("--visualizations", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--pointmap-height", type=int, default=32)
    parser.add_argument("--pointmap-width", type=int, default=64)
    parser.add_argument("--grid-x-range", default="0,3")
    parser.add_argument("--grid-y-range", default="-2,2")
    parser.add_argument("--grid-z-range", default="-0.5,2")
    parser.add_argument("--control-fps", type=float, default=30.0)
    parser.add_argument("--media-fps", type=float, default=25.0)
    parser.add_argument(
        "--camera-optical-transform",
        choices=("identity", "isaac_x_forward_from_opencv"),
        default="identity",
    )
    return parser.parse_args()


def parse_range(value: str) -> tuple[float, float]:
    values = tuple(float(item) for item in value.split(","))
    if len(values) != 2 or values[0] >= values[1]:
        raise ValueError(f"Invalid metric range: {value}")
    return values


def sample_view(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``[T,H,W,(C)]`` values with an output-size grid."""
    time, height, width = values.shape[:3]
    channels = 1 if values.ndim == 3 else values.shape[-1]
    if values.ndim == 3:
        feature = values[:, None]
    else:
        feature = values.permute(0, 3, 1, 2)
    sampled = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.permute(0, 2, 3, 1)
    return sampled[..., 0] if channels == 1 else sampled


def project_b0_points(
    points_b0: torch.Tensor,
    camera_k: torch.Tensor,
    base_from_camera: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera_from_base = invert_transform(base_from_camera)
    camera_points = torch.einsum(
        "...ij,...hwj->...hwi",
        camera_from_base[..., :3, :3],
        points_b0,
    ) + camera_from_base[..., None, None, :3, 3]
    depth = camera_points[..., 2]
    pixels_h = torch.einsum("...ij,...hwj->...hwi", camera_k, camera_points)
    pixels = pixels_h[..., :2] / pixels_h[..., 2:].clamp_min(1e-4)
    grid = torch.empty_like(pixels)
    grid[..., 0] = 2 * (pixels[..., 0] + 0.5) / width - 1
    grid[..., 1] = 2 * (pixels[..., 1] + 0.5) / height - 1
    visible = (depth > 1e-4) & (grid.abs() <= 1).all(dim=-1)
    return grid, visible


def empty_multiview_stats() -> dict[str, float]:
    result = {
        "source_valid_count": 0.0,
        "candidate_count": 0.0,
        "overlap_error_sum_m": 0.0,
    }
    for threshold in OVERLAP_THRESHOLDS_M:
        result[f"within_{threshold:.2f}m_count"] = 0.0
    return result


def add_stats(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def multiview_stats(
    pointmap: torch.Tensor,
    confidence: torch.Tensor,
    camera_k: torch.Tensor,
    base_from_camera: torch.Tensor,
    image_size: tuple[int, int],
    grid_ranges: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    time_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    """Measure independent depth agreement after head/wrist reprojection."""
    if time_indices is not None:
        pointmap = pointmap[time_indices]
        confidence = confidence[time_indices]
        camera_k = camera_k[time_indices]
        base_from_camera = base_from_camera[time_indices]
    height, width = pointmap.shape[-2:]
    render_k = scale_intrinsics(camera_k, image_size, (height, width))
    points = pointmap.permute(0, 1, 3, 4, 2)
    result = empty_multiview_stats()
    x_range, y_range, z_range = grid_ranges
    for source_view, target_view in ((0, 1), (1, 0)):
        source_points = points[:, source_view]
        source_confidence = confidence[:, source_view]
        source_inside = points_in_metric_grid(
            source_points, x_range, y_range, z_range
        )
        grid, projection_visible = project_b0_points(
            source_points,
            render_k[:, target_view],
            base_from_camera[:, target_view],
            height,
            width,
        )
        sampled_target = sample_view(points[:, target_view], grid)
        sampled_confidence = sample_view(confidence[:, target_view], grid)
        source_valid = (source_confidence > 0) & source_inside
        candidate = (
            source_valid
            & projection_visible
            & (sampled_confidence > 0)
        )
        error = torch.linalg.vector_norm(
            sampled_target - source_points, dim=-1
        )
        result["source_valid_count"] += float(source_valid.sum())
        result["candidate_count"] += float(candidate.sum())
        result["overlap_error_sum_m"] += float(
            torch.where(candidate, error, torch.zeros_like(error)).sum()
        )
        for threshold in OVERLAP_THRESHOLDS_M:
            result[f"within_{threshold:.2f}m_count"] += float(
                (candidate & (error <= threshold)).sum()
            )
    return result


def finalize_multiview(stats: dict[str, float]) -> dict[str, float]:
    source = max(1.0, stats["source_valid_count"])
    candidates = max(1.0, stats["candidate_count"])
    result = dict(stats)
    result["projection_candidate_ratio"] = stats["candidate_count"] / source
    result["candidate_mean_overlap_error_m"] = (
        stats["overlap_error_sum_m"] / candidates
    )
    for threshold in OVERLAP_THRESHOLDS_M:
        key = f"within_{threshold:.2f}m_count"
        result[f"within_{threshold:.2f}m_candidate_ratio"] = (
            stats[key] / candidates
        )
        result[f"within_{threshold:.2f}m_source_ratio"] = stats[key] / source
    return result


def right_handed_axis_rotations() -> list[tuple[str, torch.Tensor]]:
    rotations = []
    identity = torch.eye(3)
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = torch.zeros(3, 3)
            for optical_axis, pose_axis in enumerate(permutation):
                matrix[pose_axis, optical_axis] = signs[optical_axis]
            if torch.linalg.det(matrix) < 0.5:
                continue
            if torch.equal(matrix, identity):
                name = "identity"
            elif torch.equal(matrix, torch.diag(torch.tensor([1.0, -1.0, -1.0]))):
                name = "opencv_to_opengl_diag_1_-1_-1"
            else:
                encoded = "_".join(
                    f"{int(torch.argmax(matrix[:, column].abs()))}"
                    f"{'p' if matrix[:, column].sum() > 0 else 'n'}"
                    for column in range(3)
                )
                name = f"axis_{encoded}"
            transform = torch.eye(4)
            transform[:3, :3] = matrix
            rotations.append((name, transform))
    return rotations


def init_coverage(time: int, views: int) -> dict[str, np.ndarray]:
    names = (
        "valid",
        "inside",
        "x_negative",
        "x_above",
        "y_below",
        "y_above",
        "z_below",
        "z_above",
    )
    return {name: np.zeros((time, views), dtype=np.float64) for name in names}


def update_coverage(
    accumulator: dict[str, np.ndarray],
    pointmap: torch.Tensor,
    confidence: torch.Tensor,
    grid_ranges: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
) -> None:
    points = pointmap.permute(0, 1, 3, 4, 2)
    valid = confidence > 0
    x_range, y_range, z_range = grid_ranges
    conditions = {
        "valid": valid,
        "inside": valid
        & points_in_metric_grid(points, x_range, y_range, z_range),
        "x_negative": valid & (points[..., 0] < x_range[0]),
        "x_above": valid & (points[..., 0] > x_range[1]),
        "y_below": valid & (points[..., 1] < y_range[0]),
        "y_above": valid & (points[..., 1] > y_range[1]),
        "z_below": valid & (points[..., 2] < z_range[0]),
        "z_above": valid & (points[..., 2] > z_range[1]),
    }
    for name, mask in conditions.items():
        accumulator[name] += mask.sum(dim=(-2, -1)).cpu().numpy()


def finalize_coverage(
    accumulator: dict[str, np.ndarray],
) -> dict[str, object]:
    valid = accumulator["valid"]
    result: dict[str, object] = {
        "overall": {},
        "by_view": {},
        "by_time_offset": [],
    }
    for name, values in accumulator.items():
        if name == "valid":
            continue
        result["overall"][f"{name}_ratio"] = float(
            values.sum() / max(1.0, valid.sum())
        )
    for view, view_name in enumerate(VIEW_NAMES):
        result["by_view"][view_name] = {
            f"{name}_ratio": float(
                values[:, view].sum() / max(1.0, valid[:, view].sum())
            )
            for name, values in accumulator.items()
            if name != "valid"
        }
    for offset in range(valid.shape[0]):
        entry: dict[str, float | int] = {"offset": offset}
        for name, values in accumulator.items():
            if name == "valid":
                entry["valid_count"] = int(values[offset].sum())
            else:
                entry[f"{name}_ratio"] = float(
                    values[offset].sum() / max(1.0, valid[offset].sum())
                )
        result["by_time_offset"].append(entry)
    return result


def save_projection_visualization(
    path: Path,
    sample: dict[str, torch.Tensor],
    grid_ranges: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    time_index: int = 16,
) -> None:
    pointmap = sample["pseudo_pointmap_b0"]
    confidence = sample["pointmap_valid"]
    camera_k = sample["camera_K"]
    transforms = sample["T_b0_camera"]
    point_height, point_width = pointmap.shape[-2:]
    image_size = tuple(sample["video"].shape[-2:])
    render_k = scale_intrinsics(
        camera_k, image_size, (point_height, point_width)
    )
    source_points = pointmap[time_index, 0].permute(1, 2, 0)[None]
    target_points = pointmap[time_index, 1].permute(1, 2, 0)[None]
    grid, visible = project_b0_points(
        source_points,
        render_k[time_index, 1][None],
        transforms[time_index, 1][None],
        point_height,
        point_width,
    )
    sampled_target = sample_view(target_points, grid)[0]
    sampled_confidence = sample_view(
        confidence[time_index, 1][None], grid
    )[0]
    error = torch.linalg.vector_norm(
        sampled_target - source_points[0], dim=-1
    )
    inside = points_in_metric_grid(
        source_points[0], *grid_ranges
    )
    candidate = (
        visible[0]
        & inside
        & (confidence[time_index, 0] > 0)
        & (sampled_confidence > 0)
    )

    source_rgb = sample["video"][time_index, 0].permute(1, 2, 0).numpy()
    target_rgb = sample["video"][time_index, 1].permute(1, 2, 0).numpy()
    normalized = grid[0]
    u = (normalized[..., 0] + 1) * target_rgb.shape[1] / 2 - 0.5
    v = (normalized[..., 1] + 1) * target_rgb.shape[0] / 2 - 0.5

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(source_rgb)
    axes[0, 0].set_title("Head RGB source")
    axes[0, 1].imshow(target_rgb)
    scatter = axes[0, 1].scatter(
        u[candidate].numpy(),
        v[candidate].numpy(),
        c=error[candidate].numpy(),
        s=5,
        vmin=0,
        vmax=0.5,
        cmap="turbo",
    )
    axes[0, 1].set_title("Head PointMap projected into wrist")
    figure.colorbar(scatter, ax=axes[0, 1], label="B0 overlap error (m)")
    axes[1, 0].imshow(
        torch.where(candidate, error, torch.nan).numpy(),
        vmin=0,
        vmax=0.5,
        cmap="turbo",
    )
    axes[1, 0].set_title("Overlap error on source pixel lattice")
    axes[1, 1].imshow(candidate.numpy(), cmap="gray")
    axes[1, 1].set_title("Valid cross-view candidates")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def video_path(dataset_root: Path, episode: int, key: str) -> Path:
    info = json.loads((dataset_root / "meta/info.json").read_text())
    pattern = info["video_path"]
    return dataset_root / pattern.format(
        episode_chunk=episode // 1000,
        episode_index=episode,
        video_key=key,
    )


def ffprobe_video(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)["streams"][0]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_ranges = (
        parse_range(args.grid_x_range),
        parse_range(args.grid_y_range),
        parse_range(args.grid_z_range),
    )
    image_size = (args.image_height, args.image_width)
    pointmap_size = (args.pointmap_height, args.pointmap_width)
    camera_optical_transform = (
        None
        if args.camera_optical_transform == "identity"
        else ISAAC_X_FORWARD_FROM_OPENCV
    )
    dataset = MobileManiBenchVGGTDataset(
        args.dataset_root,
        video_delta_indices=VIDEO_OFFSETS,
        image_size=image_size,
        pointmap_size=pointmap_size,
        split="all",
        camera_optical_transform=camera_optical_transform,
    )
    if not len(dataset):
        raise RuntimeError("VGGT QA dataset is empty")
    sample_count = min(args.max_samples, len(dataset))
    positions = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64)

    coverage = init_coverage(len(VIDEO_OFFSETS), len(VIEW_NAMES))
    current_multiview = empty_multiview_stats()
    task_counts: Counter[str] = Counter()
    samples: list[dict[str, torch.Tensor]] = []
    selected_episodes: list[int] = []
    timing = {
        "media_index_abs_error_sum": 0.0,
        "control_index_abs_error_sum": 0.0,
        "rows": 0,
        "relative_clock_gap_s_by_offset": np.zeros(len(VIDEO_OFFSETS)),
        "control_seek_frame_error_by_offset": np.zeros(len(VIDEO_OFFSETS)),
    }

    for sample_number, position in enumerate(positions):
        sample = dataset[int(position)]
        samples.append(sample)
        episode = int(sample["episode_index"])
        selected_episodes.append(episode)
        trajectory = dataset._trajectory(episode)
        frame = int(sample["frame_index"])
        rows = trajectory.iloc[frame + np.asarray(VIDEO_OFFSETS)]
        task_counts[str(rows.iloc[0].get("annotation.task", "unknown"))] += 1

        frame_indices = rows["frame_index"].to_numpy(dtype=np.float64)
        media_timestamp = rows["timestamp"].to_numpy(dtype=np.float64)
        control_timestamp = rows["control_timestamp"].to_numpy(dtype=np.float64)
        timing["media_index_abs_error_sum"] += float(
            np.abs(media_timestamp * args.media_fps - frame_indices).sum()
        )
        timing["control_index_abs_error_sum"] += float(
            np.abs(control_timestamp * args.control_fps - frame_indices).sum()
        )
        timing["rows"] += len(rows)
        timing["relative_clock_gap_s_by_offset"] += (
            (media_timestamp - media_timestamp[0])
            - (control_timestamp - control_timestamp[0])
        )
        timing["control_seek_frame_error_by_offset"] += (
            np.rint(
                (control_timestamp - control_timestamp[0]) * args.media_fps
            )
            - np.asarray(VIDEO_OFFSETS)
        )

        update_coverage(
            coverage,
            sample["pseudo_pointmap_b0"],
            sample["pointmap_valid"],
            grid_ranges,
        )
        add_stats(
            current_multiview,
            multiview_stats(
                sample["pseudo_pointmap_b0"],
                sample["pointmap_valid"],
                sample["camera_K"],
                sample["T_b0_camera"],
                image_size,
                grid_ranges,
            ),
        )
        if sample_number < args.visualizations:
            save_projection_visualization(
                args.output_dir / f"projection_sample_{sample_number:03d}.png",
                sample,
                grid_ranges,
            )
        print(
            f"QA sample {sample_number + 1}/{sample_count}: "
            f"episode={episode}, frame={frame}"
        )

    search_count = min(args.optical_search_samples, len(samples))
    search_time = torch.tensor((0, 8, 16, 24, 32))
    optical_results = []
    for name, optical_transform in (
        right_handed_axis_rotations() if search_count else []
    ):
        aggregate = empty_multiview_stats()
        inside_count = 0.0
        valid_count = 0.0
        for sample in samples[:search_count]:
            current_transform = sample["T_b0_camera"]
            current_pointmap = sample["pseudo_pointmap_b0"]
            current_origin = current_transform[..., :3, 3]
            current_points = current_pointmap.permute(0, 1, 3, 4, 2)
            distance = torch.linalg.vector_norm(
                current_points - current_origin[..., None, None, :],
                dim=-1,
            )
            candidate_transform = current_transform @ optical_transform
            render_k = scale_intrinsics(
                sample["camera_K"], image_size, pointmap_size
            )
            candidate_pointmap = range_to_pointmap(
                distance, render_k, candidate_transform
            ).permute(0, 1, 4, 2, 3)
            candidate_points = candidate_pointmap.permute(0, 1, 3, 4, 2)
            valid = sample["pointmap_valid"] > 0
            inside = points_in_metric_grid(
                candidate_points, *grid_ranges
            )
            valid_count += float(valid[search_time].sum())
            inside_count += float((valid[search_time] & inside[search_time]).sum())
            add_stats(
                aggregate,
                multiview_stats(
                    candidate_pointmap,
                    sample["pointmap_valid"],
                    sample["camera_K"],
                    candidate_transform,
                    image_size,
                    grid_ranges,
                    search_time,
                ),
            )
        finalized = finalize_multiview(aggregate)
        finalized["name"] = name
        finalized["matrix"] = optical_transform.tolist()
        finalized["inside_grid_ratio"] = inside_count / max(1.0, valid_count)
        optical_results.append(finalized)
    optical_results.sort(
        key=lambda item: (
            item["within_0.15m_source_ratio"],
            item["inside_grid_ratio"],
        ),
        reverse=True,
    )

    ffprobe_results = []
    for episode in list(dict.fromkeys(selected_episodes))[: args.ffprobe_episodes]:
        path = video_path(
            args.dataset_root, episode, "observation.images.head"
        )
        stream = ffprobe_video(path)
        trajectory_length = len(dataset._trajectory(episode))
        ffprobe_results.append(
            {
                "episode": episode,
                "trajectory_frames": trajectory_length,
                "video_frames": int(stream["nb_frames"]),
                "duration_s": float(stream["duration"]),
                "avg_frame_rate": stream["avg_frame_rate"],
            }
        )

    timing_rows = max(1, int(timing["rows"]))
    timing_report = {
        "control_fps": args.control_fps,
        "media_fps": args.media_fps,
        "media_timestamp_frame_index_mae": (
            timing["media_index_abs_error_sum"] / timing_rows
        ),
        "control_timestamp_frame_index_mae": (
            timing["control_index_abs_error_sum"] / timing_rows
        ),
        "relative_clock_gap_s_by_offset": (
            timing["relative_clock_gap_s_by_offset"] / sample_count
        ).tolist(),
        "control_timestamp_seek_frame_error_by_offset": (
            timing["control_seek_frame_error_by_offset"] / sample_count
        ).tolist(),
        "clip_media_duration_s": VIDEO_OFFSETS[-1] / args.media_fps,
        "clip_control_duration_s": VIDEO_OFFSETS[-1] / args.control_fps,
        "clip_clock_gap_s": VIDEO_OFFSETS[-1]
        * (1 / args.media_fps - 1 / args.control_fps),
        "current_loader_alignment_authority": "media timestamp / frame index",
        "ffprobe": ffprobe_results,
    }
    report = {
        "dataset_root": str(args.dataset_root),
        "samples": sample_count,
        "task_sample_counts": dict(task_counts),
        "calibration": dataset.calibration,
        "camera_optical_transform": {
            "name": args.camera_optical_transform,
            "matrix": dataset.camera_optical_transform.tolist(),
        },
        "grid_ranges": {
            "x": grid_ranges[0],
            "y": grid_ranges[1],
            "z": grid_ranges[2],
        },
        "b0_coverage": finalize_coverage(coverage),
        "multiview_current_transform": finalize_multiview(
            current_multiview
        ),
        "optical_transform_search": {
            "samples": search_count,
            "time_offsets": search_time.tolist(),
            "ranking_metric": "within_0.15m_source_ratio",
            "top_candidates": optical_results[:8],
            "identity": (
                next(
                    item
                    for item in optical_results
                    if item["name"] == "identity"
                )
                if optical_results
                else None
            ),
        },
        "timing": timing_report,
        "interpretation_limits": [
            "The depth source is lossy H.264 pseudo-range.",
            "Intrinsics and optical-frame transforms are nominal_unverified.",
            "Optical-axis search is diagnostic and cannot replace calibration.",
        ],
    }
    output_path = args.output_dir / "qa_report.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote QA report to {output_path}")


if __name__ == "__main__":
    main()
