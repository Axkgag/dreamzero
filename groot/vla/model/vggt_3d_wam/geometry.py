"""Geometry primitives shared by the VGGT dataset and model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def euler_rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Convert roll-pitch-yaw angles to ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    roll, pitch, yaw = rpy.unbind(dim=-1)
    cr, sr = roll.cos(), roll.sin()
    cp, sp = pitch.cos(), pitch.sin()
    cy, sy = yaw.cos(), yaw.sin()
    matrix = torch.empty(*rpy.shape[:-1], 3, 3, dtype=rpy.dtype, device=rpy.device)
    matrix[..., 0, 0] = cy * cp
    matrix[..., 0, 1] = cy * sp * sr - sy * cr
    matrix[..., 0, 2] = cy * sp * cr + sy * sr
    matrix[..., 1, 0] = sy * cp
    matrix[..., 1, 1] = sy * sp * sr + cy * cr
    matrix[..., 1, 2] = sy * sp * cr - cy * sr
    matrix[..., 2, 0] = -sp
    matrix[..., 2, 1] = cp * sr
    matrix[..., 2, 2] = cp * cr
    return matrix


def pose_rpy_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    """Build a homogeneous transform from ``[..., xyz, roll, pitch, yaw]``."""
    transform = torch.zeros(
        *pose.shape[:-1], 4, 4, dtype=pose.dtype, device=pose.device
    )
    transform[..., :3, :3] = euler_rpy_to_matrix(pose[..., 3:6])
    transform[..., :3, 3] = pose[..., :3]
    transform[..., 3, 3] = 1
    return transform


def invert_transform(transform: torch.Tensor) -> torch.Tensor:
    """Invert a rigid homogeneous transform without a generic matrix inverse."""
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    inverse = torch.zeros_like(transform)
    inverse[..., :3, :3] = rotation.transpose(-1, -2)
    inverse[..., :3, 3] = -(rotation.transpose(-1, -2) @ translation[..., None])[
        ..., 0
    ]
    inverse[..., 3, 3] = 1
    return inverse


def scale_intrinsics(
    intrinsics: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Scale pinhole intrinsics between image resolutions."""
    source_h, source_w = source_size
    target_h, target_w = target_size
    result = intrinsics.clone()
    result[..., 0, :] *= target_w / source_w
    result[..., 1, :] *= target_h / source_h
    result[..., 2, :] = intrinsics[..., 2, :]
    return result


def camera_rays(
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return unit camera rays with shape ``[..., H, W, 3]``."""
    y, x = torch.meshgrid(
        torch.arange(height, dtype=intrinsics.dtype, device=intrinsics.device),
        torch.arange(width, dtype=intrinsics.dtype, device=intrinsics.device),
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    rays = torch.einsum("...ij,hwj->...hwi", torch.linalg.inv(intrinsics), pixels)
    return F.normalize(rays, dim=-1)


def rays_in_frame(
    intrinsics: torch.Tensor,
    frame_from_camera: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct ray origins and directions in a target coordinate frame."""
    camera_directions = camera_rays(intrinsics, height, width)
    rotation = frame_from_camera[..., :3, :3]
    directions = torch.einsum(
        "...ij,...hwj->...hwi", rotation, camera_directions
    )
    directions = F.normalize(directions, dim=-1)
    origins = frame_from_camera[..., None, None, :3, 3].expand_as(directions)
    return origins, directions


def range_to_pointmap(
    distance: torch.Tensor,
    intrinsics: torch.Tensor,
    frame_from_camera: torch.Tensor,
) -> torch.Tensor:
    """Lift camera-ray range into a point map in ``frame_from_camera`` coordinates."""
    height, width = distance.shape[-2:]
    origins, directions = rays_in_frame(
        intrinsics, frame_from_camera, height, width
    )
    return origins + distance[..., None] * directions


def project_points(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    frame_from_camera: torch.Tensor,
    image_size: tuple[int, int],
    min_depth: float = 1e-4,
    valid_image_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project points to a padded canvas while masking its invalid border."""
    camera_from_frame = invert_transform(frame_from_camera)
    camera_points = torch.einsum(
        "...ij,nj->...ni", camera_from_frame[..., :3, :3], points
    ) + camera_from_frame[..., None, :3, 3]
    depth = camera_points[..., 2]
    pixels_h = torch.einsum("...ij,...nj->...ni", intrinsics, camera_points)
    pixels = pixels_h[..., :2] / pixels_h[..., 2:].clamp_min(min_depth)
    height, width = image_size
    normalized = torch.empty_like(pixels)
    normalized[..., 0] = 2 * (pixels[..., 0] + 0.5) / width - 1
    normalized[..., 1] = 2 * (pixels[..., 1] + 0.5) / height - 1
    valid_height, valid_width = valid_image_size or image_size
    if valid_height > height or valid_width > width:
        raise ValueError(
            "valid_image_size cannot exceed the projected image canvas"
        )
    visible = (
        (depth > min_depth)
        & (pixels[..., 0] >= -0.5)
        & (pixels[..., 0] <= valid_width - 0.5)
        & (pixels[..., 1] >= -0.5)
        & (pixels[..., 1] <= valid_height - 0.5)
    )
    return normalized, visible


def metric_grid(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    grid_size: tuple[int, int, int],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return voxel centers ordered as ``[z, y, x]`` for 5D grid sampling."""
    z_size, y_size, x_size = grid_size

    def centers(bounds: tuple[float, float], count: int) -> torch.Tensor:
        step = (bounds[1] - bounds[0]) / count
        return torch.linspace(
            bounds[0] + 0.5 * step,
            bounds[1] - 0.5 * step,
            count,
            device=device,
            dtype=dtype,
        )

    z = centers(z_range, z_size)
    y = centers(y_range, y_size)
    x = centers(x_range, x_size)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)


def normalize_metric_points(
    points: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> torch.Tensor:
    """Map XYZ metric coordinates to the ``[-1, 1]`` grid-sample cube."""
    bounds = torch.tensor(
        [x_range, y_range, z_range], dtype=points.dtype, device=points.device
    )
    return 2 * (points - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1


def points_in_metric_grid(
    points: torch.Tensor,
    x_range: tuple[float, float] | list[float],
    y_range: tuple[float, float] | list[float],
    z_range: tuple[float, float] | list[float],
) -> torch.Tensor:
    """Return whether XYZ points lie inside the inclusive metric-grid bounds."""
    lower = points.new_tensor(
        [x_range[0], y_range[0], z_range[0]]
    )
    upper = points.new_tensor(
        [x_range[1], y_range[1], z_range[1]]
    )
    return ((points >= lower) & (points <= upper)).all(dim=-1)


def sinusoidal_encoding(values: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Encode arbitrary coordinates with fixed multi-frequency sinusoids."""
    if output_dim < 2 * values.shape[-1]:
        raise ValueError("output_dim is too small for the coordinate dimension")
    frequencies = max(1, math.ceil(output_dim / (2 * values.shape[-1])))
    scales = 2 ** torch.arange(
        frequencies, dtype=values.dtype, device=values.device
    )
    encoded = values[..., None] * scales
    encoded = torch.cat((encoded.sin(), encoded.cos()), dim=-1).flatten(-2)
    if encoded.shape[-1] < output_dim:
        encoded = F.pad(encoded, (0, output_dim - encoded.shape[-1]))
    return encoded[..., :output_dim]
