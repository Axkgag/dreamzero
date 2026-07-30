"""Rank-zero visual diagnostics for VGGT encoder-decoder training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


CONTACT_SHEET_COLUMNS = (
    "RGB GT",
    "RGB reconstruction",
    "PointMap GT range",
    "PointMap predicted range",
)


def _selected_indices(length: int, maximum: int) -> list[int]:
    if maximum <= 0 or length <= maximum:
        return list(range(length))
    return np.linspace(0, length - 1, maximum, dtype=np.int64).tolist()


def _tensor_image(value: torch.Tensor) -> np.ndarray:
    return (
        value.detach()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def _scalar_losses(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().float().cpu())
        for key, value in outputs.items()
        if torch.is_tensor(value) and value.ndim == 0
    }


def _set_equal_3d_limits(axis, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    lower = np.nanpercentile(points, 1, axis=0)
    upper = np.nanpercentile(points, 99, axis=0)
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) / 2, 0.25)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def save_vggt_visualization(
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    output_dir: str | Path,
    *,
    split: str,
    step: int,
    sample_index: int = 0,
    max_time_steps: int = 4,
    max_views: int = 2,
    scatter_max_points: int = 4096,
    confidence_threshold: float = 0.01,
    log_to_wandb: bool = False,
) -> dict[str, Path]:
    """Save synchronized reconstruction, geometry, and loss diagnostics."""
    root = (
        Path(output_dir)
        / "visualizations"
        / split
        / f"step_{step:08d}"
        / f"sample_{sample_index:03d}"
    )
    root.mkdir(parents=True, exist_ok=True)

    video = inputs["video"][0].detach().float().cpu() / 255
    reconstruction = (
        outputs["reconstructed_video"][0].detach().float().cpu() + 1
    ) * 0.5
    target_pointmap = inputs["pseudo_pointmap_b0"][0].detach().float().cpu()
    predicted_pointmap = outputs["predicted_pointmap_b0"][0].detach().float().cpu()
    confidence = inputs["pointmap_valid"][0].detach().float().cpu()
    base_from_camera = inputs["T_b0_camera"][0].detach().float().cpu()
    time_indices = _selected_indices(video.shape[0], max_time_steps)
    view_indices = _selected_indices(video.shape[1], max_views)
    rows = [(time, view) for time in time_indices for view in view_indices]

    pointmap_h, pointmap_w = target_pointmap.shape[-2:]
    rgb_small = F.interpolate(
        video.reshape(-1, 3, *video.shape[-2:]),
        size=(pointmap_h, pointmap_w),
        mode="bilinear",
        align_corners=False,
    ).reshape(video.shape[0], video.shape[1], 3, pointmap_h, pointmap_w)

    figure, axes = plt.subplots(
        len(rows),
        len(CONTACT_SHEET_COLUMNS),
        figsize=(12, max(3.0, 2.7 * len(rows))),
        dpi=110,
        squeeze=False,
    )
    for column, title in enumerate(CONTACT_SHEET_COLUMNS):
        axes[0, column].set_title(title, fontsize=10)

    for row, (time, view) in enumerate(rows):
        gt_rgb = _tensor_image(video[time, view])
        pred_rgb = _tensor_image(reconstruction[time, view])
        origin = base_from_camera[time, view, :3, 3, None, None]
        gt_range = torch.linalg.vector_norm(
            target_pointmap[time, view] - origin, dim=0
        ).numpy()
        pred_range = torch.linalg.vector_norm(
            predicted_pointmap[time, view] - origin, dim=0
        ).numpy()

        axes[row, 0].imshow(gt_rgb)
        axes[row, 1].imshow(pred_rgb)
        axes[row, 2].imshow(gt_range, cmap="viridis", vmin=0, vmax=5)
        axes[row, 3].imshow(pred_range, cmap="viridis", vmin=0, vmax=5)
        axes[row, 0].set_ylabel(f"t={time}, view={view}", fontsize=9)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    losses = _scalar_losses(outputs)
    figure.tight_layout()
    contact_sheet_path = root / "reconstruction_pointmap.png"
    figure.savefig(contact_sheet_path, bbox_inches="tight")
    plt.close(figure)

    pointmap_scatter_path = root / "pointmap_3d_scatter_b0.png"
    time, view = rows[0]
    valid = confidence[time, view] > confidence_threshold
    gt_points = target_pointmap[time, view].permute(1, 2, 0)[valid].numpy()
    pred_points = predicted_pointmap[time, view].permute(1, 2, 0)[valid].numpy()
    colors = rgb_small[time, view].permute(1, 2, 0)[valid].numpy().clip(0, 1)
    if len(gt_points) > scatter_max_points:
        keep = np.linspace(
            0, len(gt_points) - 1, scatter_max_points, dtype=np.int64
        )
        gt_points, pred_points, colors = (
            gt_points[keep],
            pred_points[keep],
            colors[keep],
        )

    point_figure = plt.figure(figsize=(10, 5), dpi=120)
    all_points = np.concatenate((gt_points, pred_points), axis=0)
    for index, (title, points) in enumerate(
        (("PointMap GT in B0", gt_points), ("PointMap prediction in B0", pred_points)),
        start=1,
    ):
        axis = point_figure.add_subplot(1, 2, index, projection="3d")
        if len(points):
            marker_size = max(4.0, min(20.0, 2000.0 / len(points)))
            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                s=marker_size,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.15,
            )
        _set_equal_3d_limits(axis, all_points)
        axis.set_title(f"{title} ({len(points)} valid pixels)")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.view_init(elev=25, azim=-60)
    point_figure.tight_layout()
    point_figure.savefig(pointmap_scatter_path, bbox_inches="tight")
    plt.close(point_figure)

    metadata_path = root / "metrics.json"
    metadata = {
        "split": split,
        "step": int(step),
        "sample_index": int(sample_index),
        "video_shape": list(inputs["video"].shape),
        "pointmap_shape": list(inputs["pseudo_pointmap_b0"].shape),
        "visualized_time_indices": time_indices,
        "visualized_view_indices": view_indices,
        "valid_confidence_sum": float(confidence.sum()),
        "losses": losses,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    paths = {
        "contact_sheet": contact_sheet_path,
        "pointmap_3d_scatter": pointmap_scatter_path,
        "metadata": metadata_path,
    }
    if log_to_wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        f"{split}/reconstruction_pointmap": wandb.Image(
                            str(contact_sheet_path)
                        ),
                        f"{split}/pointmap_3d_scatter_b0": wandb.Image(
                            str(pointmap_scatter_path)
                        ),
                    },
                    step=step,
                )
        except Exception as error:
            print(f"VGGT visualization W&B logging failed: {error}")
    return paths
