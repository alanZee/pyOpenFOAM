"""Tests for MulticomponentMixtureEnhanced4 (v5).

Tests cover:
- Soret flux computation
- Dufour heat flux computation
- Wilke viscosity computation
- Custom Soret/Dufour coefficients
"""

import pytest
import torch

from pyfoam.multiphase.multicomponent_mixture_enhanced_4 import (
    MulticomponentMixtureEnhanced4,
)
from pyfoam.multiphase.multicomponent_mixture_enhanced_3 import (
    MulticomponentMixtureEnhanced3,
)


class TestMulticomponentMixtureEnhanced4:
    """Tests for MulticomponentMixtureEnhanced4."""

    def test_inherits_from_v4(self):
        mix = MulticomponentMixtureEnhanced4(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        assert isinstance(mix, MulticomponentMixtureEnhanced3)

    def test_default_soret_dufour(self):
        mix = MulticomponentMixtureEnhanced4(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        assert mix.soret_coeff == [0.0, 0.0]
        assert mix.dufour_coeff == [0.0, 0.0]

    def test_custom_soret_dufour(self):
        mix = MulticomponentMixtureEnhanced4(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
            soret_coeff=[0.01, 0.02],
            dufour_coeff=[0.005, 0.01],
        )
        assert mix.soret_coeff == [0.01, 0.02]
        assert mix.dufour_coeff == [0.005, 0.01]

    def test_soret_flux_shape(self):
        mix = MulticomponentMixtureEnhanced4(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
            D=[2.1e-5, 2.1e-5],
            soret_coeff=[0.01, 0.02],
        )
        n_cells = 5
        Y = torch.tensor([[0.79, 0.21]] * n_cells, dtype=torch.float64)
        T = torch.full((n_cells,), 300.0, dtype=torch.float64)
        grad_T = torch.randn(n_cells, 3, dtype=torch.float64) * 100

        flux = mix.soret_flux(Y, T, grad_T)
        assert flux.shape == (n_cells, 2, 3)

    def test_dufour_heat_flux_shape(self):
        mix = MulticomponentMixtureEnhanced4(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
            D=[2.1e-5, 2.1e-5],
            dufour_coeff=[0.005, 0.01],
        )
        n_cells = 5
        Y = torch.tensor([[0.79, 0.21]] * n_cells, dtype=torch.float64)
        T = torch.full((n_cells,), 300.0, dtype=torch.float64)
        grad_Y = torch.randn(n_cells, 2, 3, dtype=torch.float64) * 0.1

        q = mix.dufour_heat_flux(Y, T, grad_Y)
        assert q.shape == (n_cells, 3)

    def test_wilke_viscosity_shape(self):
        mix = MulticomponentMixtureEnhanced4(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        n_cells = 5
        Y = torch.tensor([[0.79, 0.21]] * n_cells, dtype=torch.float64)
        T = torch.full((n_cells,), 300.0, dtype=torch.float64)

        mu_mix = mix.wilke_viscosity(Y, T)
        assert mu_mix.shape == (n_cells,)
        assert (mu_mix > 0).all()

    def test_repr(self):
        mix = MulticomponentMixtureEnhanced4(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
            soret_coeff=[0.01, 0.0],
        )
        r = repr(mix)
        assert "Enhanced4" in r
        assert "has_soret=True" in r
