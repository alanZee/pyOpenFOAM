"""Tests for v3 enhanced mapped flow rate boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_flow_rate_3 import MappedFlowRate3BC


class TestMappedFlowRate3BC:
    """Test the mappedFlowRate3 boundary condition."""

    def test_registration(self):
        assert "mappedFlowRate3" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "mappedFlowRate3", simple_patch,
            {"massFlowRate": 1.0, "rho": 1.0},
        )
        assert isinstance(bc, MappedFlowRate3BC)

    def test_type_name(self, simple_patch):
        bc = MappedFlowRate3BC(simple_patch)
        assert bc.type_name == "mappedFlowRate3"

    def test_default_properties(self, simple_patch):
        bc = MappedFlowRate3BC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.0)
        assert bc.profile_exponent == pytest.approx(7.0)
        assert bc.hydraulic_diameter == pytest.approx(0.1)
        assert bc.beta == pytest.approx(0.1)
        assert bc.Re_ref == pytest.approx(1e4)
        assert bc.n_corr == 3

    def test_custom_properties(self, simple_patch):
        bc = MappedFlowRate3BC(simple_patch, {
            "massFlowRate": 2.0, "rho": 1.2, "profileExponent": 5.0,
            "hydraulicDiameter": 0.2, "beta": 0.2, "ReRef": 5e3, "nCorr": 5,
        })
        assert bc.mass_flow_rate == pytest.approx(2.0)
        assert bc.rho == pytest.approx(1.2)
        assert bc.profile_exponent == pytest.approx(5.0)
        assert bc.hydraulic_diameter == pytest.approx(0.2)
        assert bc.beta == pytest.approx(0.2)
        assert bc.Re_ref == pytest.approx(5e3)
        assert bc.n_corr == 5

    def test_apply_default_skip(self, simple_patch):
        bc = MappedFlowRate3BC(simple_patch, {"massFlowRate": 3.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # Should produce non-zero velocity along -normal direction
        assert torch.any(field[10:13] != 0)
        # All velocities should be in -x direction (normal is +x), zero or negative
        assert torch.all(field[10:13, 0] <= 0)

    def test_apply_with_reynolds_adaptation(self, simple_patch):
        bc = MappedFlowRate3BC(simple_patch, {
            "massFlowRate": 1.0, "rho": 1.0, "profileExponent": 7.0,
            "beta": 0.1, "ReRef": 1e4, "nCorr": 3,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, velocity=velocity, nu=1e-5)

        # Should still produce valid velocity
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MappedFlowRate3BC(simple_patch, {"massFlowRate": 1.0, "rho": 1.0})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        assert torch.any(field[5:8] != 0)

    def test_iterative_correction_conserves_mass(self, simple_patch):
        bc = MappedFlowRate3BC(simple_patch, {
            "massFlowRate": 5.0, "rho": 1.0, "profileExponent": 0.0, "nCorr": 5,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # With exponent=0, uniform distribution: m_dot = rho * sum(u * area)
        areas = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        u_mag = field[10:13, 0].abs()
        m_dot = 1.0 * (u_mag * areas).sum()
        assert m_dot == pytest.approx(5.0, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = MappedFlowRate3BC(simple_patch, {"massFlowRate": 1.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
