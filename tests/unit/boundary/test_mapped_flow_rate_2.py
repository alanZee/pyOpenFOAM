"""Tests for enhanced mapped flow rate boundary condition (v2)."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_flow_rate_2 import MappedFlowRate2BC


class TestMappedFlowRate2BC:
    """Test the mappedFlowRate2 boundary condition."""

    def test_registration(self):
        assert "mappedFlowRate2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "mappedFlowRate2", simple_patch,
            {"massFlowRate": 2.0, "rho": 1.2},
        )
        assert isinstance(bc, MappedFlowRate2BC)

    def test_type_name(self, simple_patch):
        bc = MappedFlowRate2BC(simple_patch)
        assert bc.type_name == "mappedFlowRate2"

    def test_default_properties(self, simple_patch):
        bc = MappedFlowRate2BC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.0)
        assert bc.profile_exponent == pytest.approx(0.0)
        assert bc.hydraulic_diameter == pytest.approx(0.1)

    def test_custom_properties(self, simple_patch):
        bc = MappedFlowRate2BC(simple_patch, {
            "massFlowRate": 3.0,
            "rho": 1.5,
            "profileExponent": 7.0,
            "hydraulicDiameter": 0.2,
        })
        assert bc.mass_flow_rate == pytest.approx(3.0)
        assert bc.rho == pytest.approx(1.5)
        assert bc.profile_exponent == pytest.approx(7.0)
        assert bc.hydraulic_diameter == pytest.approx(0.2)

    def test_apply_uniform_profile(self, simple_patch):
        """With profileExponent=0, should behave like base MappedFlowRate."""
        bc = MappedFlowRate2BC(simple_patch, {
            "massFlowRate": 3.0,
            "rho": 1.0,
            "profileExponent": 0.0,
        })

        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # U_n = massFlowRate / (rho * totalArea) = 3.0 / (1.0 * 3.0) = 1.0
        # Velocity in -x direction = (-1.0, 0, 0)
        assert field[10, 0] == pytest.approx(-1.0)
        assert field[10, 1] == pytest.approx(0.0)

    def test_apply_with_profile(self, simple_patch):
        """With profileExponent > 0, velocity varies across faces."""
        bc = MappedFlowRate2BC(simple_patch, {
            "massFlowRate": 3.0,
            "rho": 1.0,
            "profileExponent": 2.0,
        })

        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # First face (r_frac=0) should have higher velocity than last (r_frac=1)
        assert abs(field[10, 0]) > abs(field[12, 0])

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MappedFlowRate2BC(simple_patch, {"massFlowRate": 6.0, "rho": 1.0})

        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=2)

        # Should write to indices 2, 3, 4
        assert field[2, 0] != 0.0

    def test_matrix_contributions(self, simple_patch):
        bc = MappedFlowRate2BC(simple_patch, {"massFlowRate": 3.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # u_n = 3.0 / (1.0 * 3.0) = 1.0
        # coeff = delta * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
