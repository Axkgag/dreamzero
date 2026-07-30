from __future__ import annotations

import unittest

import torch

from groot.vla.model.vggt_3d_wam.geometry import (
    invert_transform,
    points_in_metric_grid,
    pose_rpy_to_matrix,
    project_points,
    range_to_pointmap,
)


class VGGTGeometryTest(unittest.TestCase):
    def test_points_in_metric_grid_uses_inclusive_xyz_bounds(self) -> None:
        points = torch.tensor(
            [
                [0.0, -2.0, -0.5],
                [3.0, 2.0, 2.0],
                [-0.01, 0.0, 0.0],
                [1.0, 2.01, 0.0],
            ]
        )
        inside = points_in_metric_grid(
            points,
            (0.0, 3.0),
            (-2.0, 2.0),
            (-0.5, 2.0),
        )
        torch.testing.assert_close(
            inside, torch.tensor([True, True, False, False])
        )

    def test_transform_inverse(self) -> None:
        pose = torch.tensor([0.2, -0.5, 1.0, 0.1, -0.2, 0.3])
        transform = pose_rpy_to_matrix(pose)
        torch.testing.assert_close(
            invert_transform(transform) @ transform,
            torch.eye(4),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_range_pointmap_projects_back_to_pixels(self) -> None:
        intrinsics = torch.tensor(
            [[20.0, 0.0, 1.5], [0.0, 20.0, 1.5], [0.0, 0.0, 1.0]]
        )
        transform = torch.eye(4)
        distance = torch.full((4, 4), 2.0)
        pointmap = range_to_pointmap(distance, intrinsics, transform)
        grid, visible = project_points(
            pointmap.reshape(-1, 3),
            intrinsics,
            transform,
            (4, 4),
        )
        expected_y, expected_x = torch.meshgrid(
            torch.linspace(-0.75, 0.75, 4),
            torch.linspace(-0.75, 0.75, 4),
            indexing="ij",
        )
        expected = torch.stack((expected_x, expected_y), dim=-1).reshape(-1, 2)
        torch.testing.assert_close(grid, expected, atol=1e-5, rtol=1e-5)
        self.assertTrue(visible.all())

    def test_projection_masks_bottom_right_padding_strip(self) -> None:
        intrinsics = torch.eye(3)
        transform = torch.eye(4)
        points = torch.tensor(
            [
                [319.0, 159.0, 1.0],
                [320.0, 159.0, 1.0],
                [319.0, 160.0, 1.0],
            ]
        )
        _, padded_visible = project_points(
            points,
            intrinsics,
            transform,
            (168, 322),
        )
        _, original_visible = project_points(
            points,
            intrinsics,
            transform,
            (168, 322),
            valid_image_size=(160, 320),
        )
        torch.testing.assert_close(
            padded_visible, torch.tensor([True, True, True])
        )
        torch.testing.assert_close(
            original_visible, torch.tensor([True, False, False])
        )


if __name__ == "__main__":
    unittest.main()
