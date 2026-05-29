"""Tests for enhanced flow rate inlet velocity boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.flow_rate_inlet_velocity_2 import FlowRateInletVelocity2BC


class TestFlowRateInletVelocity2BC:
    """Test the flowRateInletVelocity2 boundary condition."""

    def test_registration(self):
        assert "flowRateInletVelocity2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "flowRateInletVelocity2", simple_patch,
            {"volumetricFlowRate": 0.003},
        )
        assert isinstance(bc, FlowRateInletVelocity2BC)

    def test_type_name(self, simple_patch):
        bc = FlowRateInletVelocity2BC(simple_patch)
        assert bc.type_name == "flowRateInletVelocity2"

    def test_default_properties(self, simple_patch):
        bc = FlowRateInletVelocity2BC(simple_patch)
        assert bc.volumetric_flow_rate == pytest.approx(0.0)
        assert bc.rho == pytest.approx(1.0)
        assert bc.exponent == pytest.approx(7.0)

    def test_volumetric_flow_rate(self, simple_patch):
        bc = FlowRateInletVelocity2BC(simple_patch, {
            "volumetricFlowRate": 0.003,
        })
        assert bc.volumetric_flow_rate == pytest.approx(0.003)

    def test_mass_flow_rate(self, simple_patch):
        bc = FlowRateInletVelocity2BC(simple_patch, {
            "massFlowRate": 0.003675, "rho": 1.225,
        })
        expected = 0.003675 / 1.225
        assert bc.volumetric_flow_rate == pytest.approx(expected)

    def test_custom_exponent(self, simple_patch):
        bc = FlowRateInletVelocity2BC(simple_patch, {
            "volumetricFlowRate": 0.001, "exponent": 2,
        })
        assert bc.exponent == pytest.approx(2.0)

    def test_apply_produces_velocity(self, simple_patch):
        """Should produce non-zero velocity at centre faces."""
        bc = FlowRateInletVelocity2BC(simple_patch, {
            "volumetricFlowRate": 0.003,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # Power-law profile (exponent=7) concentrates velocity at centre
        # Centre face (index 11 = face[1]) should have max velocity
        assert field[11, 0] > 0.0
        # y and z components should be zero (face normal is +x)
        assert field[11, 1] == pytest.approx(0.0, abs=1e-12)
        assert field[11, 2] == pytest.approx(0.0, abs=1e-12)

    def test_apply_zero_flow_rate(self, simple_patch):
        """Zero flow rate should produce zero velocity."""
        bc = FlowRateInletVelocity2BC(simple_patch)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        assert torch.allclose(field[10], torch.zeros(3, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = FlowRateInletVelocity2BC(simple_patch, {
            "volumetricFlowRate": 0.003,
        })
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        # Centre face (index 6 = face[1]) should have non-zero velocity
        assert field[6, 0] > 0.0

    def test_profile_shape(self, simple_patch):
        """Power-law profile should have maximum at centre."""
        bc = FlowRateInletVelocity2BC(simple_patch, {
            "volumetricFlowRate": 0.003, "exponent": 2,
        })
        profile = bc._compute_profile()
        # With 3 faces, centre face (index 1) should have highest velocity
        assert profile[1] >= profile[0]
        assert profile[1] >= profile[2]

    def test_matrix_contributions(self, simple_patch):
        bc = FlowRateInletVelocity2BC(simple_patch, {
            "volumetricFlowRate": 0.003,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert (diag > 0).all()
