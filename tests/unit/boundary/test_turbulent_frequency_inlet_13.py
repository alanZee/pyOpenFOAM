"""Tests for v13 enhanced turbulent frequency inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_frequency_inlet_13 import TurbulentFrequencyInlet13BC


class TestTurbulentFrequencyInlet13BC:
    """Test the turbulentFrequencyInlet13 boundary condition."""

    def test_registration(self):
        assert "turbulentFrequencyInlet13" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentFrequencyInlet13", simple_patch,
            {"mixingLength": 0.01},
        )
        assert isinstance(bc, TurbulentFrequencyInlet13BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentFrequencyInlet13BC(simple_patch)
        assert bc.type_name == "turbulentFrequencyInlet13"

    def test_default_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet13BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.comp_coeff == pytest.approx(0.1)
        assert bc.wp_coeff == pytest.approx(0.02)
        assert bc.wall_fluct_coeff == pytest.approx(0.5)
        assert bc.y_plus_fluct == pytest.approx(15.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet13BC(simple_patch, {
            "wpCoeff": 0.05, "wallFluctCoeff": 1.0, "yPlusFluct": 20.0,
        })
        assert bc.wp_coeff == pytest.approx(0.05)
        assert bc.wall_fluct_coeff == pytest.approx(1.0)
        assert bc.y_plus_fluct == pytest.approx(20.0)

    def test_apply_basic(self, simple_patch):
        bc = TurbulentFrequencyInlet13BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_k_and_nu(self, simple_patch):
        bc = TurbulentFrequencyInlet13BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert torch.all(torch.isfinite(field))
        assert torch.all(field[10:13] > 0)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentFrequencyInlet13BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_compressibility(self, simple_patch):
        bc = TurbulentFrequencyInlet13BC(simple_patch, {"compCoeff": 0.2})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5, c=343.0)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_pressure_gradient(self, simple_patch):
        bc = TurbulentFrequencyInlet13BC(simple_patch, {"pgCoeff": 0.1})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        pg = torch.tensor([100.0, -50.0, 200.0], dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5, velocity=velocity, pressure_gradient=pg)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentFrequencyInlet13BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentFrequencyInlet13BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(diag > 0)
