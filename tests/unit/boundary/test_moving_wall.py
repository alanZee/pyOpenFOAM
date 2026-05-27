"""Tests for movingWall boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.moving_wall import MovingWallBC


class TestMovingWallBC:
    """Test the movingWall boundary condition."""

    def test_registration(self):
        assert "movingWall" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "movingWall", simple_patch,
            {"velocity": [1, 0, 0], "omega": 10.0},
        )
        assert isinstance(bc, MovingWallBC)

    def test_type_name(self, simple_patch):
        bc = MovingWallBC(simple_patch)
        assert bc.type_name == "movingWall"

    def test_default_properties(self, simple_patch):
        bc = MovingWallBC(simple_patch)
        assert torch.allclose(bc.velocity, torch.zeros(3, dtype=torch.float64))
        assert bc.omega == pytest.approx(0.0)
        assert torch.allclose(bc.origin, torch.zeros(3, dtype=torch.float64))

    def test_translation_only(self, simple_patch):
        """Pure translation: U = (1, 2, 3), no rotation."""
        bc = MovingWallBC(simple_patch, {
            "velocity": [1.0, 2.0, 3.0],
            "omega": 0.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)
        assert torch.allclose(field[11], expected)
        assert torch.allclose(field[12], expected)

    def test_rotation_only(self, simple_patch):
        """Pure rotation: omega=1, axis=z, origin=(0,0,0).

        Face centres are at (0,0,0), (1,0,0), (2,0,0).
        v = omega * z_hat x r = (0,0,1) x (x,0,0) = (0, x, 0)
        """
        bc = MovingWallBC(simple_patch, {
            "velocity": [0.0, 0.0, 0.0],
            "origin": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "omega": 1.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # Face 0 at x=0 => v=(0,0,0), face 1 at x=1 => v=(0,1,0), face 2 at x=2 => v=(0,2,0)
        assert torch.allclose(field[10], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64))

    def test_combined_translation_rotation(self, simple_patch):
        """Translation + rotation: U_trans + omega x r."""
        bc = MovingWallBC(simple_patch, {
            "velocity": [1.0, 0.0, 0.0],
            "origin": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "omega": 2.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # Face 0 at x=0: (1,0,0) + 2*(0,0,1)x(0,0,0) = (1,0,0)
        # Face 1 at x=1: (1,0,0) + 2*(0,0,1)x(1,0,0) = (1,2,0)
        # Face 2 at x=2: (1,0,0) + 2*(0,0,1)x(2,0,0) = (1,4,0)
        assert torch.allclose(field[10], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([1.0, 4.0, 0.0], dtype=torch.float64))

    def test_axis_normalised(self, simple_patch):
        """Non-unit axis should be normalised."""
        bc = MovingWallBC(simple_patch, {
            "axis": [0.0, 0.0, 5.0],
            "omega": 1.0,
        })
        expected_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        assert torch.allclose(bc.axis, expected_axis)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MovingWallBC(simple_patch, {"velocity": [0.0, 5.0, 0.0]})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=7)

        expected = torch.tensor([0.0, 5.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[7], expected)

    def test_matrix_contributions(self, simple_patch):
        bc = MovingWallBC(simple_patch, {"velocity": [1.0, 0.0, 0.0]})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = 2.0 * 1.0 = 2.0, velocity_x = 1.0 => source = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(
            source, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        )
