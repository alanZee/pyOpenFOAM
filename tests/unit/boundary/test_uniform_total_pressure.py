"""Tests for uniform total pressure boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.uniform_total_pressure import UniformTotalPressureBC


class TestUniformTotalPressureBC:
    """Test the uniformTotalPressure boundary condition."""

    def test_registration(self):
        assert "uniformTotalPressure" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "uniformTotalPressure", simple_patch, {"p0": 101325.0},
        )
        assert isinstance(bc, UniformTotalPressureBC)

    def test_type_name(self, simple_patch):
        bc = UniformTotalPressureBC(simple_patch)
        assert bc.type_name == "uniformTotalPressure"

    def test_default_p0(self, simple_patch):
        bc = UniformTotalPressureBC(simple_patch)
        assert bc.p0 == pytest.approx(101325.0)

    def test_custom_p0(self, simple_patch):
        bc = UniformTotalPressureBC(simple_patch, {"p0": 200000.0})
        assert bc.p0 == pytest.approx(200000.0)

    def test_apply_without_velocity(self, simple_patch):
        """Without velocity, sets boundary to p0."""
        bc = UniformTotalPressureBC(simple_patch, {"p0": 1e5})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(
            field[simple_patch.face_indices],
            torch.full((3,), 1e5, dtype=torch.float64),
        )

    def test_apply_with_velocity(self, simple_patch):
        """With velocity, subtracts dynamic pressure."""
        bc = UniformTotalPressureBC(simple_patch, {"p0": 1e5})
        velocity = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, rho=1.2)
        # p = 1e5 - 0.5*1.2*|U|^2
        expected = 1e5 - 0.5 * 1.0 * 1.2  # face 0: |U|^2=1
        assert field[10] == pytest.approx(expected)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = UniformTotalPressureBC(simple_patch, {"p0": 5e4})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=3)
        assert torch.allclose(
            field[3:6], torch.full((3,), 5e4, dtype=torch.float64),
        )

    def test_matrix_contributions(self, simple_patch):
        bc = UniformTotalPressureBC(simple_patch, {"p0": 1e5})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([2e5, 2e5, 2e5], dtype=torch.float64))
