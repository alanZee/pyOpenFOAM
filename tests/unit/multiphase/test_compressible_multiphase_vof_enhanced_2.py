"""Tests for CompressibleMultiphaseVoFEnhanced2.

Tests cover:
- Temperature-dependent viscosity (Sutherland)
- Mixture speed of sound
- Total enthalpy
- Coupled EOS iteration
- Properties
"""

import pytest
import torch

from pyfoam.multiphase.compressible_multiphase_vof_enhanced_2 import (
    CompressibleMultiphaseVoFEnhanced2,
)
from pyfoam.multiphase.compressible_multiphase_vof_enhanced import (
    CompressibleMultiphaseVoFEnhanced,
)


class TestCompressibleMultiphaseVoFEnhanced2:
    """Tests for CompressibleMultiphaseVoFEnhanced2."""

    def _make_model(self, **kwargs):
        defaults = dict(
            phase_names=["gas", "liquid"],
            eos_type=["perfectGas", "incompressible"],
            rho_ref=[1.225, 998.0],
            mu=[1.8e-5, 1.002e-3],
            R=[287.0, None],
            gamma=[1.4, None],
        )
        defaults.update(kwargs)
        return CompressibleMultiphaseVoFEnhanced2(**defaults)

    def test_inherits_from_enhanced(self):
        model = self._make_model()
        assert isinstance(model, CompressibleMultiphaseVoFEnhanced)

    def test_default_params(self):
        model = self._make_model()
        assert model.viscosity_model == "constant"
        assert model.S_sutherland == pytest.approx(110.4)
        assert model.n_eos_iter == 3

    def test_sutherland_viscosity(self):
        model = self._make_model(viscosity_model="sutherland")
        alphas = torch.tensor([[0.5], [0.8]], dtype=torch.float64)
        T = torch.tensor([300.0, 500.0], dtype=torch.float64)
        mu_mix = model.mixture_viscosity(alphas, T)
        assert mu_mix.shape == (2,)
        assert (mu_mix > 0).all()

    def test_constant_viscosity(self):
        model = self._make_model(viscosity_model="constant")
        alphas = torch.tensor([[0.5]], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        mu_mix = model.mixture_viscosity(alphas, T)
        assert mu_mix.shape == (1,)
        assert (mu_mix > 0).all()

    def test_sutherland_temperature_dependence(self):
        """Higher temperature should give higher viscosity for Sutherland model."""
        model = self._make_model(viscosity_model="sutherland")
        alphas = torch.tensor([[0.5]], dtype=torch.float64)
        T_low = torch.tensor([300.0], dtype=torch.float64)
        T_high = torch.tensor([600.0], dtype=torch.float64)
        mu_low = model.mixture_viscosity(alphas, T_low)
        mu_high = model.mixture_viscosity(alphas, T_high)
        assert float(mu_high[0].item()) > float(mu_low[0].item())

    def test_mixture_speed_of_sound(self):
        model = self._make_model()
        alphas = torch.tensor([[0.5]], dtype=torch.float64)
        p = torch.tensor([101325.0], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        a_m = model.mixture_speed_of_sound(alphas, p, T)
        assert a_m.shape == (1,)
        assert float(a_m[0].item()) > 0

    def test_mixture_total_enthalpy(self):
        model = self._make_model()
        alphas = torch.tensor([[0.5]], dtype=torch.float64)
        p = torch.tensor([101325.0], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        U_mag = torch.tensor([10.0], dtype=torch.float64)
        h = model.mixture_total_enthalpy(alphas, p, T, U_mag)
        assert h.shape == (1,)
        assert float(h[0].item()) > 0

    def test_iterate_eos_coupled(self):
        model = self._make_model(n_eos_iter=2)
        alphas = torch.tensor([[0.5]], dtype=torch.float64)
        p = torch.tensor([101325.0], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        rho_target = torch.tensor([500.0], dtype=torch.float64)
        p_new, T_new = model.iterate_eos_coupled(alphas, p, T, rho_target)
        assert p_new.shape == (1,)
        assert T_new.shape == (1,)
        assert float(p_new[0].item()) > 0
        assert float(T_new[0].item()) > 0

    def test_repr(self):
        model = self._make_model(viscosity_model="sutherland")
        r = repr(model)
        assert "CompressibleMultiphaseVoFEnhanced2" in r
        assert "sutherland" in r
