"""Tests for v5 enhanced mapped flow rate boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_flow_rate_5 import MappedFlowRate5BC


class TestMappedFlowRate5BC:
    """Test the mappedFlowRate5 boundary condition."""

    def test_registration(self):
        assert "mappedFlowRate5" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "mappedFlowRate5", simple_patch,
            {"massFlowRate": 1.0, "rho": 1.0},
        )
        assert isinstance(bc, MappedFlowRate5BC)

    def test_type_name(self, simple_patch):
        bc = MappedFlowRate5BC(simple_patch)
        assert bc.type_name == "mappedFlowRate5"

    def test_default_properties(self, simple_patch):
        bc = MappedFlowRate5BC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.0)
        assert bc.profile_exponent == pytest.approx(7.0)
        assert bc.hydraulic_diameter == pytest.approx(0.1)
        assert bc.beta == pytest.approx(0.1)
        assert bc.Re_ref == pytest.approx(1e4)
        assert bc.n_corr == 3
        assert bc.Cp == pytest.approx(1005.0)
        assert bc.gamma_Cp == pytest.approx(0.0)
        assert bc.swirl_ratio == pytest.approx(0.0)
        assert bc.swirl_exponent == pytest.approx(1.0)

    def test_custom_properties(self, simple_patch):
        bc = MappedFlowRate5BC(simple_patch, {
            "massFlowRate": 2.0, "rho": 1.2, "profileExponent": 5.0,
            "hydraulicDiameter": 0.2, "beta": 0.2, "ReRef": 5e3, "nCorr": 5,
            "Cp": 1000.0, "gammaCp": 0.001, "swirlRatio": 0.3, "swirlExponent": 2.0,
        })
        assert bc.mass_flow_rate == pytest.approx(2.0)
        assert bc.rho == pytest.approx(1.2)
        assert bc.Cp == pytest.approx(1000.0)
        assert bc.gamma_Cp == pytest.approx(0.001)
        assert bc.swirl_ratio == pytest.approx(0.3)
        assert bc.swirl_exponent == pytest.approx(2.0)

    def test_apply_default(self, simple_patch):
        bc = MappedFlowRate5BC(simple_patch, {"massFlowRate": 3.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        assert torch.any(field[10:13] != 0)
        assert torch.all(field[10:13, 0] <= 0)

    def test_apply_with_thermal_expansion(self, simple_patch):
        bc = MappedFlowRate5BC(simple_patch, {
            "massFlowRate": 1.0, "rho": 1.0, "betaThermal": 0.001,
            "temperature": 350.0, "TRef": 300.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_variable_cp(self, simple_patch):
        bc = MappedFlowRate5BC(simple_patch, {
            "massFlowRate": 1.0, "rho": 1.0,
            "Cp": 1005.0, "gammaCp": 0.002,
            "temperature": 400.0, "TRef": 300.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_swirl(self, simple_patch):
        bc = MappedFlowRate5BC(simple_patch, {
            "massFlowRate": 1.0, "rho": 1.0, "swirlRatio": 0.5, "swirlExponent": 1.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # Swirl should produce tangential velocity components
        assert torch.any(field[10:13, 1] != 0) or torch.any(field[10:13, 2] != 0)

    def test_apply_with_reynolds_adaptation(self, simple_patch):
        bc = MappedFlowRate5BC(simple_patch, {
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

        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MappedFlowRate5BC(simple_patch, {"massFlowRate": 1.0, "rho": 1.0})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        assert torch.any(field[5:8] != 0)

    def test_iterative_correction_conserves_mass(self, simple_patch):
        bc = MappedFlowRate5BC(simple_patch, {
            "massFlowRate": 5.0, "rho": 1.0, "profileExponent": 0.0, "nCorr": 5,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        areas = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        u_mag = field[10:13, 0].abs()
        m_dot = 1.0 * (u_mag * areas).sum()
        assert m_dot == pytest.approx(5.0, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = MappedFlowRate5BC(simple_patch, {"massFlowRate": 1.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
