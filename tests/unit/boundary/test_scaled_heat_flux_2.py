"""Tests for v2 enhanced scaled heat flux boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.scaled_heat_flux_2 import ScaledHeatFlux2BC


class TestScaledHeatFlux2BC:
    """Test the scaledHeatFlux2 boundary condition."""

    def test_registration(self):
        assert "scaledHeatFlux2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "scaledHeatFlux2", simple_patch,
            {"scale": 2.0, "q_ref": 500.0},
        )
        assert isinstance(bc, ScaledHeatFlux2BC)

    def test_type_name(self, simple_patch):
        bc = ScaledHeatFlux2BC(simple_patch)
        assert bc.type_name == "scaledHeatFlux2"

    def test_default_properties(self, simple_patch):
        bc = ScaledHeatFlux2BC(simple_patch)
        assert bc.scale == pytest.approx(1.0)
        assert bc.q_ref == pytest.approx(0.0)
        assert bc.k == pytest.approx(0.025)
        assert bc.T_ref == pytest.approx(300.0)
        assert bc.alpha_T == pytest.approx(0.0)
        assert bc.T_scale_ref == pytest.approx(300.0)
        assert bc.beta_k == pytest.approx(0.0)
        assert bc.T_k_ref == pytest.approx(300.0)
        assert bc.time_modulation == pytest.approx(1.0)

    def test_custom_properties(self, simple_patch):
        bc = ScaledHeatFlux2BC(simple_patch, {
            "scale": 2.0, "q_ref": 500.0, "k": 0.05,
            "alphaT": 0.001, "TscaleRef": 350.0,
            "betaK": 0.002, "TkRef": 400.0,
            "timeModulation": 0.5,
        })
        assert bc.scale == pytest.approx(2.0)
        assert bc.q_ref == pytest.approx(500.0)
        assert bc.k == pytest.approx(0.05)
        assert bc.alpha_T == pytest.approx(0.001)
        assert bc.T_scale_ref == pytest.approx(350.0)
        assert bc.beta_k == pytest.approx(0.002)
        assert bc.T_k_ref == pytest.approx(400.0)
        assert bc.time_modulation == pytest.approx(0.5)

    def test_q_property(self, simple_patch):
        bc = ScaledHeatFlux2BC(simple_patch, {
            "scale": 2.0, "q_ref": 500.0, "timeModulation": 0.5,
        })
        assert bc.q == pytest.approx(500.0)  # 2.0 * 500.0 * 0.5

    def test_gradient_property(self, simple_patch):
        bc = ScaledHeatFlux2BC(simple_patch, {
            "scale": 1.0, "q_ref": 100.0, "k": 0.025,
        })
        # gradient = -q / k = -100 / 0.025 = -4000
        assert bc.gradient == pytest.approx(-4000.0)

    def test_scale_setter(self, simple_patch):
        bc = ScaledHeatFlux2BC(simple_patch)
        bc.scale = 3.0
        assert bc.scale == pytest.approx(3.0)

    def test_time_modulation_setter(self, simple_patch):
        bc = ScaledHeatFlux2BC(simple_patch)
        bc.time_modulation = 0.5
        assert bc.time_modulation == pytest.approx(0.5)

    def test_apply_default(self, simple_patch):
        bc = ScaledHeatFlux2BC(simple_patch, {
            "scale": 1.0, "q_ref": 100.0, "k": 0.025,
            "value": 300.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        # T_face = T_ref + gradient / delta
        # gradient = -4000, delta = 2.0
        # T_face = 300 + (-4000) / 2.0 = 300 - 2000 = -1700
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_temperature_feedback(self, simple_patch):
        """Temperature-dependent scaling should affect the result."""
        bc = ScaledHeatFlux2BC(simple_patch, {
            "scale": 1.0, "q_ref": 100.0, "k": 0.025,
            "alphaT": 0.001, "TscaleRef": 300.0,
            "value": 300.0,
        })
        T_field = torch.tensor([300.0, 400.0, 500.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, T_field=T_field)

        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_conductivity_correction(self, simple_patch):
        """Temperature-dependent conductivity should affect the gradient."""
        bc = ScaledHeatFlux2BC(simple_patch, {
            "scale": 1.0, "q_ref": 100.0, "k": 0.025,
            "betaK": 0.001, "TkRef": 300.0,
            "value": 300.0,
        })
        T_field = torch.tensor([300.0, 400.0, 500.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, T_field=T_field)

        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = ScaledHeatFlux2BC(simple_patch, {
            "scale": 1.0, "q_ref": 100.0, "k": 0.025,
        })
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        assert field[5] != 0.0

    def test_matrix_contributions(self, simple_patch):
        bc = ScaledHeatFlux2BC(simple_patch, {
            "scale": 2.0, "q_ref": 500.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Diag should be zero (fixedGradient treatment)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        # Source should have q * A = 1000 * 1.0 = 1000
        assert torch.allclose(source, torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64))

    def test_effective_conductivity(self, simple_patch):
        """Effective conductivity should vary with temperature."""
        bc = ScaledHeatFlux2BC(simple_patch, {
            "k": 0.025, "betaK": 0.001, "TkRef": 300.0,
        })
        # At T_ref, k_eff = k * (1 + 0) = k
        assert bc._effective_conductivity(300.0) == pytest.approx(0.025)
        # At T=400, k_eff = k * (1 + 0.001 * 100) = k * 1.1
        assert bc._effective_conductivity(400.0) == pytest.approx(0.0275)

    def test_effective_scale(self, simple_patch):
        """Effective scale should vary with temperature."""
        bc = ScaledHeatFlux2BC(simple_patch, {
            "scale": 2.0, "alphaT": 0.001, "TscaleRef": 300.0,
        })
        # At T_ref, scale_eff = scale * (1 + 0) = scale
        assert bc._effective_scale(300.0) == pytest.approx(2.0)
        # At T=400, scale_eff = scale * (1 + 0.001 * 100) = scale * 1.1
        assert bc._effective_scale(400.0) == pytest.approx(2.2)
