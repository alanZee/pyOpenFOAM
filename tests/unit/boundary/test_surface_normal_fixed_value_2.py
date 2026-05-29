"""Tests for enhanced surface normal fixed value boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.surface_normal_fixed_value_2 import SurfaceNormalFixedValue2BC


class TestSurfaceNormalFixedValue2BC:
    """Test the surfaceNormalFixedValue2 boundary condition."""

    def test_registration(self):
        assert "surfaceNormalFixedValue2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "surfaceNormalFixedValue2", simple_patch,
            {"value": 5.0, "tangentialFraction": 0.0},
        )
        assert isinstance(bc, SurfaceNormalFixedValue2BC)

    def test_type_name(self, simple_patch):
        bc = SurfaceNormalFixedValue2BC(simple_patch, {"value": 1.0})
        assert bc.type_name == "surfaceNormalFixedValue2"

    def test_default_properties(self, simple_patch):
        bc = SurfaceNormalFixedValue2BC(simple_patch, {"value": 3.0})
        assert bc.magnitude.item() == pytest.approx(3.0)
        assert bc.tangential_fraction == pytest.approx(0.0)

    def test_custom_tangential_fraction(self, simple_patch):
        bc = SurfaceNormalFixedValue2BC(simple_patch, {
            "value": 1.0, "tangentialFraction": 0.5,
        })
        assert bc.tangential_fraction == pytest.approx(0.5)

    def test_apply_vector_field_pure_normal(self, simple_patch):
        """Pure normal (tangentialFraction=0): velocity = magnitude * normal."""
        bc = SurfaceNormalFixedValue2BC(simple_patch, {"value": 5.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # normal = (1,0,0), magnitude = 5 -> velocity = (5,0,0)
        assert field[10, 0] == pytest.approx(5.0)
        assert field[10, 1] == pytest.approx(0.0)
        assert field[10, 2] == pytest.approx(0.0)

    def test_apply_vector_field_with_tangential(self, simple_patch):
        """With tangentialFraction > 0, should preserve some interior tangential velocity."""
        bc = SurfaceNormalFixedValue2BC(simple_patch, {
            "value": 5.0, "tangentialFraction": 1.0,
        })

        field = torch.zeros((15, 3), dtype=torch.float64)
        # Owner cell 0 has velocity (0, 3, 4) -> tangential = (0, 3, 4)
        field[0] = torch.tensor([0.0, 3.0, 4.0], dtype=torch.float64)
        interior = torch.tensor([[0.0, 3.0, 4.0]], dtype=torch.float64).expand(3, 3).contiguous()

        bc.apply(field, interior_velocity=interior)

        # normal = (1,0,0), magnitude = 5
        # normal_vel = (5,0,0)
        # interior = (0,3,4), projection on normal = (0,0,0)
        # tangential = (0,3,4)
        # result = (5,0,0) + 1.0 * (0,3,4) = (5,3,4)
        assert field[10, 0] == pytest.approx(5.0)
        assert field[10, 1] == pytest.approx(3.0)
        assert field[10, 2] == pytest.approx(4.0)

    def test_apply_scalar_field(self, simple_patch):
        """Scalar field should store magnitude."""
        bc = SurfaceNormalFixedValue2BC(simple_patch, {"value": 7.5})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        assert field[10] == pytest.approx(7.5)
        assert field[11] == pytest.approx(7.5)
        assert field[12] == pytest.approx(7.5)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = SurfaceNormalFixedValue2BC(simple_patch, {"value": 2.0})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        assert field[5] == pytest.approx(2.0)
        assert field[6] == pytest.approx(2.0)
        assert field[7] == pytest.approx(2.0)

    def test_magnitude_setter(self, simple_patch):
        bc = SurfaceNormalFixedValue2BC(simple_patch, {"value": 1.0})
        bc.magnitude = 10.0
        assert bc.magnitude.item() == pytest.approx(10.0)

    def test_matrix_contributions(self, simple_patch):
        bc = SurfaceNormalFixedValue2BC(simple_patch, {"value": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * magnitude = 2.0 * 5.0 = 10.0
        assert torch.allclose(source, torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64))
