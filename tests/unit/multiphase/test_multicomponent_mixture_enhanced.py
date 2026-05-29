"""Tests for MulticomponentMixtureEnhanced.

Tests cover:
- Effective diffusivity (molecular + turbulent)
- Species diffusion flux
- Redlich-Kister activity coefficients
- Reaction rates interface
"""

import pytest
import torch

from pyfoam.multiphase.multicomponent_mixture import MulticomponentMixture
from pyfoam.multiphase.multicomponent_mixture_enhanced import (
    MulticomponentMixtureEnhanced,
)


class TestMulticomponentMixtureEnhanced:
    """Tests for MulticomponentMixtureEnhanced."""

    def _make_mix(self, **kwargs):
        defaults = dict(
            species=["N2", "O2", "H2O"],
            M=[28.014e-3, 32.0e-3, 18.015e-3],
            rho=[1.165, 1.331, 0.804],
            mu=[1.76e-5, 2.04e-5, 0.96e-5],
            Cp=[1040.0, 919.0, 2080.0],
        )
        defaults.update(kwargs)
        return MulticomponentMixtureEnhanced(**defaults)

    def test_inherits_from_base(self):
        mix = self._make_mix()
        assert isinstance(mix, MulticomponentMixture)

    def test_default_diffusivity(self):
        mix = self._make_mix()
        assert mix.D == [2e-5, 2e-5, 2e-5]
        assert mix.Sc_t == [0.7, 0.7, 0.7]

    def test_custom_diffusivity(self):
        mix = self._make_mix(D=[1e-5, 2e-5, 3e-5], Sc_t=[0.5, 0.7, 0.9])
        assert mix.D == [1e-5, 2e-5, 3e-5]
        assert mix.Sc_t == [0.5, 0.7, 0.9]

    def test_molecular_diffusivity_only(self):
        mix = self._make_mix(D=[1e-5, 2e-5, 3e-5])
        Y = torch.tensor([[0.3, 0.3, 0.4]], dtype=torch.float64)
        D_eff = mix.effective_diffusivity(Y)
        assert D_eff.shape == (1, 3)
        assert torch.allclose(
            D_eff[0], torch.tensor([1e-5, 2e-5, 3e-5], dtype=torch.float64),
        )

    def test_turbulent_diffusivity(self):
        mix = self._make_mix(D=[1e-5, 1e-5, 1e-5], Sc_t=[1.0, 1.0, 1.0])
        Y = torch.tensor([[0.3, 0.3, 0.4]], dtype=torch.float64)
        mu_t = torch.tensor([0.01], dtype=torch.float64)
        D_eff = mix.effective_diffusivity(Y, mu_t=mu_t)
        assert D_eff.shape == (1, 3)
        # D_eff = D_mol + mu_t / (rho_m * Sc_t)
        rho_m = mix.mixture_density(Y)
        D_turb = mu_t / rho_m
        expected = 1e-5 + D_turb
        assert torch.allclose(D_eff[0], expected.expand(3), rtol=1e-3)

    def test_species_diffusion_flux(self):
        mix = self._make_mix(D=[1e-5, 2e-5, 3e-5])
        Y = torch.tensor([[0.3, 0.3, 0.4]], dtype=torch.float64)
        grad_Y = torch.tensor([[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]],
                              dtype=torch.float64)
        flux = mix.species_diffusion_flux(Y, grad_Y)
        assert flux.shape == (1, 3, 3)
        # J_i = -D_eff * grad_Y_i
        assert torch.allclose(flux[0, 0, 0], torch.tensor(-1e-5 * 0.1, dtype=torch.float64))

    def test_activity_coefficient_ideal(self):
        """Ideal mixture: all gamma = 1."""
        mix = self._make_mix()
        Y = torch.tensor([[0.3, 0.3, 0.4]], dtype=torch.float64)
        gamma = mix.activity_coefficient_rk(Y)
        assert gamma.shape == (1, 3)
        assert torch.allclose(gamma, torch.ones(1, 3, dtype=torch.float64), atol=1e-10)

    def test_activity_coefficient_nonideal(self):
        """Non-ideal: A != 0 gives gamma != 1."""
        mix = self._make_mix()
        Y = torch.tensor([[0.3, 0.3, 0.4]], dtype=torch.float64)
        gamma = mix.activity_coefficient_rk(Y, A=[1.0, 0.5, 0.0])
        assert gamma.shape == (1, 3)
        # gamma[0] = exp(1.0 * (1-0.3)^2) > 1
        assert float(gamma[0, 0].item()) > 1.0

    def test_reaction_rates_zero_default(self):
        mix = self._make_mix()
        Y = torch.tensor([[0.3, 0.3, 0.4]], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        omega = mix.compute_reaction_rates(Y, T)
        assert torch.allclose(omega, torch.zeros(1, 3, dtype=torch.float64))

    def test_reaction_rates_provided(self):
        mix = self._make_mix()
        Y = torch.tensor([[0.3, 0.3, 0.4]], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        omega_in = torch.tensor([[0.1, -0.05, 0.02]], dtype=torch.float64)
        omega = mix.compute_reaction_rates(Y, T, omega_dot=omega_in)
        assert torch.allclose(omega, omega_in)

    def test_repr(self):
        mix = self._make_mix()
        r = repr(mix)
        assert "MulticomponentMixtureEnhanced" in r
        assert "N2" in r
