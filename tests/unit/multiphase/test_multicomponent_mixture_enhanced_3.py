"""Tests for MulticomponentMixtureEnhanced3 (v4).

Tests cover:
- Reaction source terms
- Wassiljewa conductivity
- Effective diffusivity with Lewis number
- Inheritance
"""

import pytest
import torch

from pyfoam.multiphase.multicomponent_mixture_enhanced_3 import (
    MulticomponentMixtureEnhanced3,
)
from pyfoam.multiphase.multicomponent_mixture_enhanced_2 import (
    MulticomponentMixtureEnhanced2,
)


def _make_mixture(**kwargs):
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
    return MulticomponentMixtureEnhanced3(**defaults)


class TestMulticomponentMixtureEnhanced3:
    """Tests for MulticomponentMixtureEnhanced3."""

    def test_inherits_from_v3(self):
        mix = _make_mixture()
        assert isinstance(mix, MulticomponentMixtureEnhanced2)

    def test_default_lewis_numbers(self):
        mix = _make_mixture()
        assert mix.Le == [1.0, 1.0, 1.0]

    def test_custom_lewis_numbers(self):
        mix = _make_mixture(Le=[1.0, 0.7, 0.8])
        assert mix.Le == [1.0, 0.7, 0.8]

    def test_invalid_lewis_numbers_length(self):
        with pytest.raises(ValueError):
            _make_mixture(Le=[1.0, 0.7])  # Length mismatch

    def test_default_reaction_rates(self):
        mix = _make_mixture()
        assert mix.reaction_rates == [0.0, 0.0, 0.0]

    def test_custom_reaction_rates(self):
        mix = _make_mixture(reaction_rates=[0.0, 0.0, -0.1])
        assert mix.reaction_rates[2] == pytest.approx(-0.1)

    def test_reaction_source_shape(self):
        mix = _make_mixture(reaction_rates=[0.0, 0.0, -0.1])
        Y = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]], dtype=torch.float64)
        rho_mix = torch.tensor([1.0, 1.0], dtype=torch.float64)
        S = mix.reaction_source(Y, rho_mix)
        assert S.shape == (2, 3)
        # Only H2O has non-zero rate
        assert float(S[0, 0].item()) == pytest.approx(0.0)
        assert float(S[0, 2].item()) != 0.0

    def test_wassiljewa_conductivity_shape(self):
        mix = _make_mixture(kappa=[0.026, 0.026, 0.018])
        Y = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        kappa = mix.wassiljewa_conductivity(Y, T)
        assert kappa.shape == (1,)
        assert float(kappa[0].item()) > 0

    def test_effective_diffusivity_shape(self):
        mix = _make_mixture(Le=[1.0, 0.7, 0.8])
        Y = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        D_eff = mix.effective_diffusivity(Y, T)
        assert D_eff.shape == (1, 3)
        assert (D_eff > 0).all()

    def test_effective_diffusivity_with_turbulent(self):
        mix = _make_mixture(Le=[1.0, 0.7, 0.8])
        Y = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
        T = torch.tensor([300.0], dtype=torch.float64)
        D_t = torch.tensor([1e-4], dtype=torch.float64)
        D_eff = mix.effective_diffusivity(Y, T, D_t)
        assert D_eff.shape == (1, 3)
        # With turbulent contribution, should be larger
        D_mol = mix.effective_diffusivity(Y, T)
        assert (D_eff >= D_mol).all()

    def test_repr(self):
        mix = _make_mixture(Le=[1.0, 0.7, 0.8], reaction_rates=[0.0, 0.0, -0.1])
        r = repr(mix)
        assert "Enhanced3" in r
        assert "has_reaction=True" in r
