"""Tests for multi-component mixture model.

Tests cover:
- MulticomponentMixture initialisation and validation
- Mass fraction normalisation
- Mole-to-mass and mass-to-mole conversion
- Mixture density, viscosity, Cp, kappa, M
- compute_all convenience method
"""

import pytest
import torch

from pyfoam.multiphase.multicomponent_mixture import MulticomponentMixture


class TestMulticomponentMixture:
    """Tests for MulticomponentMixture."""

    def test_init_two_species(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        assert mix.n_species == 2
        assert mix.species == ["N2", "O2"]

    def test_init_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="M length"):
            MulticomponentMixture(
                ["N2", "O2"],
                M=[28.0e-3],
                rho=[1.165, 1.331],
                mu=[1.76e-5, 2.04e-5],
                Cp=[1040.0, 919.0],
            )

    def test_init_empty_species_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            MulticomponentMixture([], M=[], rho=[], mu=[], Cp=[])

    def test_validate_mass_fractions(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        # Already normalised
        Y = torch.tensor([[0.79, 0.21], [0.5, 0.5]], dtype=torch.float64)
        Y_n = mix.validate_mass_fractions(Y)
        assert torch.allclose(Y_n.sum(dim=-1), torch.ones(2, dtype=torch.float64), atol=1e-6)

    def test_validate_renormalises(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        Y = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
        Y_n = mix.validate_mass_fractions(Y)
        assert torch.allclose(Y_n.sum(dim=-1), torch.ones(1, dtype=torch.float64), atol=1e-6)
        # Ratios preserved: 2/5 = 0.4, 3/5 = 0.6
        assert torch.allclose(Y_n, torch.tensor([[0.4, 0.6]], dtype=torch.float64), atol=1e-6)

    def test_mole_to_mass_and_back(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        X = torch.tensor([[0.79, 0.21]], dtype=torch.float64)
        Y = mix.mole_to_mass(X)
        assert Y.shape == (1, 2)
        assert torch.allclose(Y.sum(dim=-1), torch.ones(1, dtype=torch.float64), atol=1e-6)

        X_back = mix.mass_to_mole(Y)
        assert torch.allclose(X_back, X, atol=1e-4)

    def test_mixture_density(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        # Pure N2: rho_m = rho_N2
        Y = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        rho = mix.mixture_density(Y)
        assert torch.allclose(rho, torch.tensor([1.165], dtype=torch.float64), atol=1e-3)

    def test_mixture_viscosity(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        Y = torch.tensor([[0.79, 0.21]], dtype=torch.float64)
        mu = mix.mixture_viscosity(Y)
        expected = 0.79 * 1.76e-5 + 0.21 * 2.04e-5
        assert torch.allclose(mu, torch.tensor([expected], dtype=torch.float64), atol=1e-10)

    def test_mixture_cp(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        Y = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        cp = mix.mixture_Cp(Y)
        expected = 0.5 * 1040.0 + 0.5 * 919.0
        assert torch.allclose(cp, torch.tensor([expected], dtype=torch.float64), atol=1e-6)

    def test_mixture_kappa_from_prandtl(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        Y = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        kappa = mix.mixture_kappa(Y, Pr=0.7)
        mu = mix.mixture_viscosity(Y)
        cp = mix.mixture_Cp(Y)
        expected = mu * cp / 0.7
        assert torch.allclose(kappa, expected, atol=1e-10)

    def test_mixture_kappa_tabulated(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
            kappa=[0.026, 0.027],
        )
        Y = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        kappa = mix.mixture_kappa(Y)
        expected = 0.5 * 0.026 + 0.5 * 0.027
        assert torch.allclose(kappa, torch.tensor([expected], dtype=torch.float64), atol=1e-6)

    def test_mixture_molecular_weight(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        # Equal mass fractions
        Y = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        M_m = mix.mixture_M(Y)
        # M_m = 1 / (0.5/0.028014 + 0.5/0.032)
        expected = 1.0 / (0.5 / 0.028014 + 0.5 / 0.032)
        assert torch.allclose(M_m, torch.tensor([expected], dtype=torch.float64), rtol=1e-4)

    def test_compute_all(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        Y = torch.tensor([[0.79, 0.21]], dtype=torch.float64)
        result = mix.compute_all(Y)
        assert "rho_m" in result
        assert "mu_m" in result
        assert "Cp_m" in result
        assert "kappa_m" in result
        assert "M_m" in result

    def test_repr(self):
        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
        r = repr(mix)
        assert "MulticomponentMixture" in r
        assert "N2" in r
        assert "2" in r
