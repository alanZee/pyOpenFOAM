"""Tests for turbulent temperature coupled boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_temperature_coupled import TurbulentTemperatureCoupledBC


class TestTurbulentTemperatureCoupledBC:
    """Test the turbulentTemperatureCoupled boundary condition."""

    def test_registration(self):
        assert "turbulentTemperatureCoupled" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentTemperatureCoupled", simple_patch,
            {"T_coupled": 350.0, "Pr_t": 0.85},
        )
        assert isinstance(bc, TurbulentTemperatureCoupledBC)

    def test_type_name(self, simple_patch):
        bc = TurbulentTemperatureCoupledBC(simple_patch)
        assert bc.type_name == "turbulentTemperatureCoupled"

    def test_default_properties(self, simple_patch):
        bc = TurbulentTemperatureCoupledBC(simple_patch)
        assert bc.t_coupled == pytest.approx(300.0)
        assert bc.pr_t == pytest.approx(0.85)
        assert bc.alpha_lam == pytest.approx(2.5e-5)
        assert bc.nut_values is None

    def test_custom_properties(self, simple_patch):
        bc = TurbulentTemperatureCoupledBC(simple_patch, {
            "T_coupled": 350.0,
            "Pr_t": 0.9,
            "alpha_lam": 1e-4,
        })
        assert bc.t_coupled == pytest.approx(350.0)
        assert bc.pr_t == pytest.approx(0.9)
        assert bc.alpha_lam == pytest.approx(1e-4)

    def test_nut_values_setter(self, simple_patch):
        bc = TurbulentTemperatureCoupledBC(simple_patch)
        nut = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc.nut_values = nut
        assert bc.nut_values is not None
        assert torch.allclose(bc.nut_values, nut)

    def test_apply_no_nut(self, simple_patch):
        """Without nut values, uses only laminar diffusivity."""
        bc = TurbulentTemperatureCoupledBC(simple_patch, {
            "T_coupled": 300.0,
            "alpha_lam": 2.5e-5,
            "Pr_t": 0.85,
        })
        field = torch.full((15,), 350.0, dtype=torch.float64)
        bc.apply(field)
        # alpha_eff = alpha_lam (nut=0), grad_weight = alpha_lam * delta
        # T = (alpha_lam * delta * T_int + alpha_lam * T_coupled) / (alpha_lam * delta + alpha_lam)
        # T = (delta * T_int + T_coupled) / (delta + 1)
        delta = 2.0
        expected = (delta * 350.0 + 300.0) / (delta + 1.0)
        assert field[10] == pytest.approx(expected)

    def test_apply_with_nut(self, simple_patch):
        """With nut values, enhanced diffusivity changes the blend."""
        bc = TurbulentTemperatureCoupledBC(simple_patch, {
            "T_coupled": 300.0,
            "alpha_lam": 0.0,
            "Pr_t": 0.85,
        })
        nut = torch.tensor([0.85, 0.85, 0.85], dtype=torch.float64)
        bc.nut_values = nut
        field = torch.full((15,), 400.0, dtype=torch.float64)
        bc.apply(field)
        # alpha_eff = 0 + 0.85/0.85 = 1.0
        # grad_weight = 1.0 * 2.0 = 2.0, value_weight = 1.0
        # T = (2 * 400 + 1 * 300) / (2 + 1) = 1100/3
        expected = (2.0 * 400.0 + 300.0) / 3.0
        assert field[10] == pytest.approx(expected)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentTemperatureCoupledBC(simple_patch, {"T_coupled": 300.0})
        field = torch.full((20,), 350.0, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        delta = 2.0
        expected = (delta * 350.0 + 300.0) / (delta + 1.0)
        assert field[5] == pytest.approx(expected)

    def test_matrix_contributions_no_nut(self, simple_patch):
        bc = TurbulentTemperatureCoupledBC(simple_patch, {
            "T_coupled": 300.0,
            "alpha_lam": 2.5e-5,
        })
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        assert diag[0] > 0.0
        assert source[0] > 0.0

    def test_matrix_contributions_with_nut(self, simple_patch):
        """Matrix contributions increase with nut."""
        bc = TurbulentTemperatureCoupledBC(simple_patch, {
            "T_coupled": 300.0,
            "alpha_lam": 0.0,
            "Pr_t": 0.85,
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.nut_values = torch.tensor([0.85, 0.85, 0.85], dtype=torch.float64)
        diag_nut, _ = bc.matrix_contributions(field, n_cells=3)

        bc2 = TurbulentTemperatureCoupledBC(simple_patch, {
            "T_coupled": 300.0,
            "alpha_lam": 0.0,
            "Pr_t": 0.85,
        })
        bc2.nut_values = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        diag_no_nut, _ = bc2.matrix_contributions(field, n_cells=3)

        # With nut, diag should be larger
        assert diag_nut[0] > diag_no_nut[0]
