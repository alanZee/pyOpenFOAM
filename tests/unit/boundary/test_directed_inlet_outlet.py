"""Tests for directed inlet/outlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.directed_inlet_outlet import DirectedInletOutletBC


class TestDirectedInletOutletBC:
    """Test the directedInletOutlet boundary condition."""

    def test_registration(self):
        assert "directedInletOutlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "directedInletOutlet", simple_patch,
            {"direction": [1, 0, 0], "U_mag": 5.0},
        )
        assert isinstance(bc, DirectedInletOutletBC)

    def test_type_name(self, simple_patch):
        bc = DirectedInletOutletBC(simple_patch)
        assert bc.type_name == "directedInletOutlet"

    def test_default_direction(self, simple_patch):
        bc = DirectedInletOutletBC(simple_patch)
        assert torch.allclose(bc.direction, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    def test_custom_direction_normalised(self, simple_patch):
        bc = DirectedInletOutletBC(simple_patch, {"direction": [3, 4, 0]})
        d = bc.direction
        assert d.norm().item() == pytest.approx(1.0)

    def test_default_u_mag(self, simple_patch):
        bc = DirectedInletOutletBC(simple_patch)
        assert bc.u_mag == pytest.approx(1.0)

    def test_custom_u_mag(self, simple_patch):
        bc = DirectedInletOutletBC(simple_patch, {"U_mag": 10.0})
        assert bc.u_mag == pytest.approx(10.0)

    def test_apply_inlet_direction(self, simple_patch):
        """Inlet (no flux): velocity = U_mag * direction."""
        bc = DirectedInletOutletBC(simple_patch, {"direction": [1, 0, 0], "U_mag": 3.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        expected = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)
        assert torch.allclose(field[11], expected)
        assert torch.allclose(field[12], expected)

    def test_apply_with_negative_flux_inlet(self, simple_patch):
        """Negative flux (inward) selects inlet treatment."""
        bc = DirectedInletOutletBC(simple_patch, {"direction": [1, 0, 0], "U_mag": 2.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        flux = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64)
        bc.apply(field, flux=flux)
        assert torch.allclose(field[10], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))

    def test_apply_with_positive_flux_outlet(self, simple_patch):
        """Positive flux (outward) selects zero-gradient treatment."""
        bc = DirectedInletOutletBC(simple_patch, {"direction": [1, 0, 0], "U_mag": 2.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        field[0] = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64)
        flux = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc.apply(field, flux=flux)
        # Should copy owner values
        assert torch.allclose(field[10], torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = DirectedInletOutletBC(simple_patch, {"direction": [0, 1, 0], "U_mag": 4.0})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        expected = torch.tensor([0.0, 4.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)
        assert torch.allclose(field[6], expected)
        assert torch.allclose(field[7], expected)

    def test_matrix_contributions(self, simple_patch):
        bc = DirectedInletOutletBC(simple_patch, {"direction": [1, 0, 0], "U_mag": 5.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * (dir_x * U_mag) = 2 * (1.0 * 5.0) = 10
        assert torch.allclose(source, torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64))
