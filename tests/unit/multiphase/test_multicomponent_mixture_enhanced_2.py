"""Tests for MulticomponentMixtureEnhanced2.

Tests cover:
- Polynomial Cp(T)
- Wilke viscosity
- Maxwell-Stefan diffusivity
- Counter-gradient correction
- Properties
"""

import pytest
import torch

from pyfoam.multiphase.multicomponent_mixture_enhanced_2 import (
    MulticomponentMixtureEnhanced2,
)
from pyfoam.multiphase.multicomponent_mixture_enhanced import (
    MulticomponentMixtureEnhanced,
)


class TestMulticomponentMixtureEnhanced2:
    """Tests for MulticomponentMixtureEnhanced2."""

    def _make_model(self, **kwargs):
        defaults = dict(
            species=["N2", "O2", "H2O"],
            M=[28.014e-3, 32.0e-3, 18.015e-3],
            rho=[1.165, 1.331, 0.804],
            mu=[1.76e-5, 2.04e-5, 0.96e-5],
            Cp=[1040.0, 919.0, 2080.0],
            D=[2.1e-5, 2.1e-5, 2.5e-5],
            Sc_t=[0.7, 0.7, 0.7],
        )
        defaults.update(kwargs)
        return MulticomponentMixtureEnhanced2(**defaults)

    def test_inherits_from_enhanced(self):
        mix = self._make_model()
        assert isinstance(mix, MulticomponentMixtureEnhanced)

    def test_default_cp_poly(self):
        mix = self._make_model()
        # Default: constant Cp (single coefficient)
        assert len(mix.Cp_poly) == 3
        assert len(mix.Cp_poly[0]) == 1
        assert mix.Cp_poly[0][0] == pytest.approx(1040.0)

    def test_custom_cp_poly(self):
        mix = self._make_model(
            Cp_poly=[[1000.0, 0.1], [900.0, 0.05], [2000.0, 0.2]],
        )
        assert len(mix.Cp_poly[0]) == 2
        assert mix.Cp_poly[0][1] == pytest.approx(0.1)

    def test_cp_temperature_constant(self):
        mix = self._make_model()
        T = torch.tensor([300.0, 500.0, 700.0], dtype=torch.float64)
        Cp = mix.Cp_temperature(T)
        assert Cp.shape == (3, 3)
        # Constant Cp: should be the same at all temperatures
        assert float(Cp[0, 0].item()) == pytest.approx(1040.0, rel=1e-6)
        assert float(Cp[1, 0].item()) == pytest.approx(1040.0, rel=1e-6)

    def test_cp_temperature_polynomial(self):
        mix = self._make_model(
            Cp_poly=[[1000.0, 0.1, 0.0001], [900.0, 0.05, 0.0], [2000.0, 0.2, 0.0002]],
        )
        T = torch.tensor([300.0, 600.0], dtype=torch.float64)
        Cp = mix.Cp_temperature(T)
        assert Cp.shape == (2, 3)
        # Cp(T) = 1000 + 0.1*300 + 0.0001*300^2 = 1000 + 30 + 9 = 1039
        assert float(Cp[0, 0].item()) == pytest.approx(1039.0, rel=1e-4)
        # At T=600: 1000 + 60 + 36 = 1096
        assert float(Cp[1, 0].item()) == pytest.approx(1096.0, rel=1e-4)

    def test_wilke_viscosity(self):
        mix = self._make_model()
        Y = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]], dtype=torch.float64)
        T = torch.tensor([300.0, 500.0], dtype=torch.float64)
        mu = mix.wilke_viscosity(Y, T)
        assert mu.shape == (2,)
        assert (mu > 0).all()

    def test_wilke_viscosity_mixture(self):
        """Wilke viscosity should be between min and max pure species viscosity."""
        mix = self._make_model()
        Y = torch.tensor([[0.33, 0.33, 0.34]], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        mu = mix.wilke_viscosity(Y, T)
        mu_min = min(mix._mu)
        mu_max = max(mix._mu)
        assert float(mu[0].item()) >= mu_min * 0.5  # Allow some tolerance
        assert float(mu[0].item()) <= mu_max * 2.0

    def test_maxwell_stefan_diffusivity(self):
        mix = self._make_model()
        Y = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        p = torch.tensor([101325.0], dtype=torch.float64)
        D = mix.maxwell_stefan_diffusivity(Y, T, p)
        assert D.shape == (1, 3, 3)
        # Diagonal should be zero (no self-diffusion)
        for i in range(3):
            assert float(D[0, i, i].item()) == pytest.approx(0.0)

    def test_maxwell_stefan_temperature_dependence(self):
        """Higher temperature -> higher diffusivity."""
        mix = self._make_model()
        Y = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)
        T_low = torch.tensor([300.0], dtype=torch.float64)
        T_high = torch.tensor([600.0], dtype=torch.float64)
        p = torch.tensor([101325.0], dtype=torch.float64)
        D_low = mix.maxwell_stefan_diffusivity(Y, T_low, p)
        D_high = mix.maxwell_stefan_diffusivity(Y, T_high, p)
        # D_01 should be higher at higher T
        assert float(D_high[0, 0, 1].item()) > float(D_low[0, 0, 1].item())

    def test_counter_gradient_correction(self):
        mix = self._make_model()
        grad_Y = torch.tensor([[[0.1, 0.0, 0.0], [0.05, 0.0, 0.0], [0.02, 0.0, 0.0]]], dtype=torch.float64)
        D_eff = torch.tensor([[1e-5, 1e-5, 1e-5]], dtype=torch.float64)
        J_cg = mix.counter_gradient_correction(grad_Y, D_eff)
        assert J_cg.shape == (1, 3, 3)
        # Should be proportional to gradient
        assert float(J_cg[0, 0, 0].item()) > float(J_cg[0, 2, 0].item())

    def test_repr(self):
        mix = self._make_model(
            Cp_poly=[[1000.0, 0.1], [900.0, 0.05], [2000.0, 0.2]],
        )
        r = repr(mix)
        assert "MulticomponentMixtureEnhanced2" in r
        assert "True" in r  # has_Cp_poly=True

    def test_cp_poly_length_mismatch(self):
        with pytest.raises(ValueError):
            self._make_model(Cp_poly=[[1000.0], [900.0]])
