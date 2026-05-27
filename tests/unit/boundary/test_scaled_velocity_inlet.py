"""Tests for scaledVelocityInlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.scaled_velocity_inlet import ScaledVelocityInletBC


class TestScaledVelocityInletBC:
    """Test the scaledVelocityInlet boundary condition."""

    def test_registration(self):
        assert "scaledVelocityInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "scaledVelocityInlet", simple_patch,
            {"scale": 1.5, "U_ref": "inletRef"},
        )
        assert isinstance(bc, ScaledVelocityInletBC)

    def test_type_name(self, simple_patch):
        bc = ScaledVelocityInletBC(simple_patch, {"scale": 2.0})
        assert bc.type_name == "scaledVelocityInlet"

    def test_default_scale(self, simple_patch):
        bc = ScaledVelocityInletBC(simple_patch)
        assert bc.scale == pytest.approx(1.0)

    def test_scale_property(self, simple_patch):
        bc = ScaledVelocityInletBC(simple_patch, {"scale": 2.5})
        assert bc.scale == pytest.approx(2.5)

    def test_default_U_ref_value(self, simple_patch):
        bc = ScaledVelocityInletBC(simple_patch)
        assert torch.allclose(bc.U_ref_value, torch.zeros(3, dtype=torch.float64))

    def test_U_ref_value_from_coeffs(self, simple_patch):
        bc = ScaledVelocityInletBC(simple_patch, {
            "scale": 1.0, "U_ref_value": [1.0, 2.0, 3.0],
        })
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        assert torch.allclose(bc.U_ref_value, expected)

    def test_apply_with_U_ref_value(self, simple_patch):
        """U_ref_value = (1, 0, 0), scale = 2 => velocity = (2, 0, 0)."""
        bc = ScaledVelocityInletBC(simple_patch, {
            "scale": 2.0, "U_ref_value": [1.0, 0.0, 0.0],
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)
        assert torch.allclose(field[11], expected)
        assert torch.allclose(field[12], expected)

    def test_apply_with_ref_velocity_tensor(self, simple_patch):
        """External ref_velocity passed at apply time."""
        bc = ScaledVelocityInletBC(simple_patch, {"scale": 3.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        ref = torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=torch.float64)
        bc.apply(field, ref_velocity=ref)

        # Mean of ref = (1.0, 0.5, 0.0), scale = 3 => (3, 1.5, 0)
        expected = torch.tensor([3.0, 1.5, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = ScaledVelocityInletBC(simple_patch, {
            "scale": 1.0, "U_ref_value": [0.0, 5.0, 0.0],
        })
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        expected = torch.tensor([0.0, 5.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)

    def test_apply_zero_scale(self, simple_patch):
        """Zero scale yields zero velocity regardless of U_ref."""
        bc = ScaledVelocityInletBC(simple_patch, {
            "scale": 0.0, "U_ref_value": [100.0, 100.0, 100.0],
        })
        field = torch.ones((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_negative_scale(self, simple_patch):
        """Negative scale reverses the reference velocity."""
        bc = ScaledVelocityInletBC(simple_patch, {
            "scale": -1.0, "U_ref_value": [1.0, 0.0, 0.0],
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)

    def test_matrix_contributions(self, simple_patch):
        bc = ScaledVelocityInletBC(simple_patch, {
            "scale": 2.0, "U_ref_value": [1.0, 0.0, 0.0],
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        # velocity_x = 2.0 => source = 2.0 * 2.0 = 4.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(
            source, torch.tensor([4.0, 4.0, 4.0], dtype=torch.float64)
        )
