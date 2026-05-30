"""Tests for MulticomponentMixtureEnhanced5 (v6).

Tests cover:
- Stefan-Maxwell diffusion flux
- NRTL activity coefficients
- Mixture enthalpy
- Custom parameters
- Inheritance
"""

import pytest
import torch

from pyfoam.multiphase.multicomponent_mixture_enhanced_5 import (
    MulticomponentMixtureEnhanced5,
)
from pyfoam.multiphase.multicomponent_mixture_enhanced_4 import (
    MulticomponentMixtureEnhanced4,
)


class TestMulticomponentMixtureEnhanced5:
    """Tests for MulticomponentMixtureEnhanced5."""

    def test_inherits_from_v5(self):
        mix = MulticomponentMixtureEnhanced5(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        assert isinstance(mix, MulticomponentMixtureEnhanced4)

    def test_default_D_ij(self):
        mix = MulticomponentMixtureEnhanced5(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
            D=[2.1e-5, 2.1e-5],
        )
        D_ij = mix.D_ij
        assert len(D_ij) == 2
        assert D_ij[0][0] == pytest.approx(0.0)
        assert D_ij[0][1] == pytest.approx(2.1e-5)

    def test_custom_D_ij(self):
        mix = MulticomponentMixtureEnhanced5(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
            D_ij=[[0.0, 3e-5], [3e-5, 0.0]],
        )
        assert mix.D_ij[0][1] == pytest.approx(3e-5)

    def test_stefan_maxwell_flux_shape(self):
        mix = MulticomponentMixtureEnhanced5(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
            D=[2.1e-5, 2.1e-5],
        )
        n_cells = 5
        Y = torch.tensor([[0.79, 0.21]] * n_cells, dtype=torch.float64)
        grad_X = torch.randn(n_cells, 2, 3, dtype=torch.float64) * 0.01

        flux = mix.stefan_maxwell_flux(Y, grad_X)
        assert flux.shape == (n_cells, 2, 3)

    def test_nrtl_activity_coefficients_shape(self):
        mix = MulticomponentMixtureEnhanced5(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        n_cells = 5
        Y = torch.tensor([[0.79, 0.21]] * n_cells, dtype=torch.float64)
        T = torch.full((n_cells,), 300.0, dtype=torch.float64)

        gamma = mix.nrtl_activity_coefficients(Y, T)
        assert gamma.shape == (n_cells, 2)
        assert (gamma > 0).all()

    def test_mixture_enthalpy_shape(self):
        mix = MulticomponentMixtureEnhanced5(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
            H_ref=[0.0, 0.0],
        )
        n_cells = 5
        Y = torch.tensor([[0.79, 0.21]] * n_cells, dtype=torch.float64)
        T = torch.full((n_cells,), 300.0, dtype=torch.float64)

        h = mix.mixture_enthalpy(Y, T)
        assert h.shape == (n_cells,)

    def test_repr(self):
        mix = MulticomponentMixtureEnhanced5(
            species=["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        r = repr(mix)
        assert "Enhanced5" in r
