"""Tests for wall distance calculators."""

import pytest
import torch

from pyfoam.turbulence.wall_distance import (
    WallDistanceCalculator,
    ExactWallDistance,
    ApproximateWallDistance,
)


# ------------------------------------------------------------------
# ExactWallDistance
# ------------------------------------------------------------------


class TestExactWallDistance:
    """Test the ExactWallDistance calculator."""

    def test_basic_distance_to_flat_wall(self):
        """Cell 1 unit from a flat wall at x=0."""
        # Wall at x=0, normals face +x
        wall_centres = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        wall_normals = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        calc = ExactWallDistance(wall_centres, wall_normals)

        # Cell at x=1, y=0, z=0 => perpendicular distance = 1
        cell_centres = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        y = calc.compute(cell_centres)
        assert torch.allclose(y, torch.tensor([1.0], dtype=torch.float64), atol=1e-10)

    def test_distance_to_wall_at_angle(self):
        """Cell distance to a tilted wall."""
        # Wall at (0,0,0) with normal (0,1,0) (horizontal wall)
        wall_centres = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        wall_normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        calc = ExactWallDistance(wall_centres, wall_normals)

        # Cell at (0, 5, 0) => perpendicular distance = 5
        cell_centres = torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64)
        y = calc.compute(cell_centres)
        assert torch.allclose(y, torch.tensor([5.0], dtype=torch.float64), atol=1e-6)

    def test_multiple_cells_nearest_wall(self):
        """Multiple cells, each should find the nearest wall face."""
        # Two wall faces: one at x=0, one at x=10
        wall_centres = torch.tensor([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        wall_normals = torch.tensor([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=torch.float64)
        calc = ExactWallDistance(wall_centres, wall_normals)

        # Cells at x=1, x=9 => distances 1 and 1
        cell_centres = torch.tensor([
            [1.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ], dtype=torch.float64)
        y = calc.compute(cell_centres)
        assert torch.allclose(y, torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-10)

    def test_cell_between_two_walls(self):
        """Cell between two walls gets minimum distance."""
        wall_centres = torch.tensor([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        wall_normals = torch.tensor([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=torch.float64)
        calc = ExactWallDistance(wall_centres, wall_normals)

        # Cell at x=3 => nearest wall at x=0, distance=3
        cell_centres = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
        y = calc.compute(cell_centres)
        assert torch.allclose(y, torch.tensor([3.0], dtype=torch.float64), atol=1e-10)

    def test_n_wall_faces_property(self):
        """n_wall_faces returns the number of wall faces."""
        wall_centres = torch.zeros(5, 3, dtype=torch.float64)
        wall_normals = torch.zeros(5, 3, dtype=torch.float64)
        wall_normals[:, 0] = 1.0
        calc = ExactWallDistance(wall_centres, wall_normals)
        assert calc.n_wall_faces == 5

    def test_mismatched_sizes_raises(self):
        """Mismatched wall_centres and wall_normals raises ValueError."""
        wc = torch.zeros(3, 3, dtype=torch.float64)
        wn = torch.zeros(5, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="same number"):
            ExactWallDistance(wc, wn)

    def test_batch_large_input(self):
        """Works correctly with a larger batch of cells."""
        wall_centres = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        wall_normals = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        calc = ExactWallDistance(wall_centres, wall_normals)

        n = 200
        cc = torch.zeros(n, 3, dtype=torch.float64)
        cc[:, 0] = torch.linspace(1.0, 10.0, n)
        y = calc.compute(cc)
        assert y.shape == (n,)
        assert (y > 0).all()


# ------------------------------------------------------------------
# ApproximateWallDistance
# ------------------------------------------------------------------


class TestApproximateWallDistance:
    """Test the ApproximateWallDistance calculator."""

    def test_basic_distance_to_flat_wall(self):
        """Cell 1 unit from a flat wall."""
        wall_centres = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        calc = ApproximateWallDistance(wall_centres)

        cell_centres = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        y = calc.compute(cell_centres)
        assert torch.allclose(y, torch.tensor([1.0], dtype=torch.float64), atol=1e-10)

    def test_correction_factor(self):
        """Correction factor scales the distance."""
        wall_centres = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        calc = ApproximateWallDistance(wall_centres, correction_factor=2.0)

        cell_centres = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        y = calc.compute(cell_centres)
        assert torch.allclose(y, torch.tensor([2.0], dtype=torch.float64), atol=1e-10)

    def test_multiple_cells_nearest_wall(self):
        """Multiple cells, each should find the nearest wall face."""
        wall_centres = torch.tensor([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        calc = ApproximateWallDistance(wall_centres)

        cell_centres = torch.tensor([
            [1.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
        ], dtype=torch.float64)
        y = calc.compute(cell_centres)
        assert torch.allclose(y, torch.tensor([1.0, 1.0], dtype=torch.float64), atol=1e-10)

    def test_n_wall_faces_property(self):
        """n_wall_faces returns the number of wall faces."""
        wall_centres = torch.zeros(7, 3, dtype=torch.float64)
        calc = ApproximateWallDistance(wall_centres)
        assert calc.n_wall_faces == 7

    def test_wall_normals_optional(self):
        """Wall normals are optional for the approximate method."""
        wall_centres = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        calc = ApproximateWallDistance(wall_centres, wall_normals=None)
        cc = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
        y = calc.compute(cc)
        assert torch.allclose(y, torch.tensor([3.0], dtype=torch.float64), atol=1e-10)

    def test_distance_grows_with_cell_position(self):
        """Distance increases as cell moves further from wall."""
        wall_centres = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        calc = ApproximateWallDistance(wall_centres)

        n = 50
        cc = torch.zeros(n, 3, dtype=torch.float64)
        cc[:, 0] = torch.linspace(1.0, 10.0, n)
        y = calc.compute(cc)
        # Distance should be monotonically increasing
        assert (y[1:] >= y[:-1] - 1e-10).all()

    def test_batch_large_input(self):
        """Works correctly with a larger batch of cells."""
        wall_centres = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        calc = ApproximateWallDistance(wall_centres)

        n = 200
        cc = torch.zeros(n, 3, dtype=torch.float64)
        cc[:, 0] = torch.linspace(1.0, 10.0, n)
        y = calc.compute(cc)
        assert y.shape == (n,)
        assert (y > 0).all()
