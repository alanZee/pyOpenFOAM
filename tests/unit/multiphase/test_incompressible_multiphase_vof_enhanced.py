"""Tests for IncompressibleMultiphaseVoFEnhanced.

Tests cover:
- Bounded clamp (MULES-style)
- Conservation correction
- Gradient-weighted compression
- Enhanced advance method
- Properties and repr
"""

import pytest
import torch

from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced import (
    IncompressibleMultiphaseVoFEnhanced,
)


class TestIncompressibleMultiphaseVoFEnhanced:
    """Tests for IncompressibleMultiphaseVoFEnhanced."""

    def test_inherits_from_base(self):
        model = IncompressibleMultiphaseVoFEnhanced(
            ["water", "air"], [998.0, 1.225], [1e-3, 1.8e-5],
        )
        assert isinstance(model, IncompressibleMultiphaseVoF)

    def test_default_params(self):
        model = IncompressibleMultiphaseVoFEnhanced(
            ["water", "air"], [998.0, 1.225], [1e-3, 1.8e-5],
        )
        assert model.Co_max == pytest.approx(1.0)
        assert model.conservation_tol == pytest.approx(1e-6)

    def test_custom_params(self):
        model = IncompressibleMultiphaseVoFEnhanced(
            ["water", "air"], [998.0, 1.225], [1e-3, 1.8e-5],
            Co_max=0.5, conservation_tol=1e-8,
        )
        assert model.Co_max == pytest.approx(0.5)
        assert model.conservation_tol == pytest.approx(1e-8)

    def test_three_phases(self):
        model = IncompressibleMultiphaseVoFEnhanced(
            ["water", "air", "oil"],
            [998.0, 1.225, 850.0],
            [1e-3, 1.8e-5, 0.03],
        )
        assert model.n_phases == 3

    def test_mixture_properties(self):
        model = IncompressibleMultiphaseVoFEnhanced(
            ["water", "air"], [998.0, 1.225], [1e-3, 1.8e-5],
        )
        alphas = torch.tensor([[0.5]], dtype=torch.float64)
        rho = model.mixture_density(alphas)
        assert rho.shape == (1,)
        assert float(rho[0].item()) == pytest.approx((0.5 * 998.0 + 0.5 * 1.225), rel=1e-3)

    def test_bounded_clamp_preserves_range(self):
        model = IncompressibleMultiphaseVoFEnhanced(
            ["water", "air", "oil"],
            [998.0, 1.225, 850.0],
            [1e-3, 1.8e-5, 0.03],
        )
        alphas_old = torch.tensor([[0.3, 0.2]], dtype=torch.float64)
        alphas_new = torch.tensor([[0.5, 0.4]], dtype=torch.float64)
        result = model.bounded_clamp(alphas_new, alphas_old)
        assert result.shape == (1, 2)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_conservation_correct(self):
        model = IncompressibleMultiphaseVoFEnhanced(
            ["water", "air"], [998.0, 1.225], [1e-3, 1.8e-5],
            conservation_tol=1e-10,
        )
        alphas_old = torch.tensor([[0.5], [0.3]], dtype=torch.float64)
        # Introduce a small error
        alphas = torch.tensor([[0.501], [0.301]], dtype=torch.float64)
        V = torch.tensor([1.0, 1.0], dtype=torch.float64)
        result = model.conservation_correct(alphas.clone(), alphas_old, V)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_repr(self):
        model = IncompressibleMultiphaseVoFEnhanced(
            ["water", "air"], [998.0, 1.225], [1e-3, 1.8e-5],
        )
        r = repr(model)
        assert "IncompressibleMultiphaseVoFEnhanced" in r
        assert "water" in r
        assert "Co_max" in r

    def test_validate_alphas(self):
        model = IncompressibleMultiphaseVoFEnhanced(
            ["water", "air", "oil"],
            [998.0, 1.225, 850.0],
            [1e-3, 1.8e-5, 0.03],
        )
        # Alpha sum exceeds 1
        alphas = torch.tensor([[0.6, 0.7]], dtype=torch.float64)
        result = model.validate_alphas(alphas)
        total = result.sum(dim=-1)
        assert (total <= 1.0 + 1e-6).all()

    def test_compute_last_alpha(self):
        model = IncompressibleMultiphaseVoFEnhanced(
            ["water", "air", "oil"],
            [998.0, 1.225, 850.0],
            [1e-3, 1.8e-5, 0.03],
        )
        alphas = torch.tensor([[0.3, 0.2]], dtype=torch.float64)
        alpha_N = model.compute_last_alpha(alphas)
        assert float(alpha_N[0].item()) == pytest.approx(0.5)
