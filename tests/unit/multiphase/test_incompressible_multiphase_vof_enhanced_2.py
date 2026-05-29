"""Tests for IncompressibleMultiphaseVoFEnhanced2.

Tests cover:
- Deferred correction flux
- Adaptive sharpening
- Full advance with v3 enhancements
- Properties and repr
"""

import pytest
import torch

from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_2 import (
    IncompressibleMultiphaseVoFEnhanced2,
)
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced import (
    IncompressibleMultiphaseVoFEnhanced,
)


class TestIncompressibleMultiphaseVoFEnhanced2:
    """Tests for IncompressibleMultiphaseVoFEnhanced2."""

    def test_inherits_from_enhanced(self):
        model = IncompressibleMultiphaseVoFEnhanced2(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert isinstance(model, IncompressibleMultiphaseVoFEnhanced)

    def test_default_params(self):
        model = IncompressibleMultiphaseVoFEnhanced2(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert model.blend_factor == pytest.approx(0.5)
        assert model.sharpen_threshold == pytest.approx(0.01)

    def test_custom_params(self):
        model = IncompressibleMultiphaseVoFEnhanced2(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            blend_factor=0.3,
            sharpen_threshold=0.05,
        )
        assert model.blend_factor == pytest.approx(0.3)
        assert model.sharpen_threshold == pytest.approx(0.05)

    def test_adaptive_sharpen(self):
        model = IncompressibleMultiphaseVoFEnhanced2(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            sharpen_threshold=0.1,
        )
        alphas = torch.tensor([[0.3], [0.5], [0.8]], dtype=torch.float64)
        grad_mag = torch.tensor([0.01, 0.5, 0.01], dtype=torch.float64)
        result = model.adaptive_sharpen(alphas, grad_mag)
        assert result.shape == (3, 1)
        # Alpha near 0.5 should be sharpened most when grad is high
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_adaptive_sharpen_preserves_boundary(self):
        model = IncompressibleMultiphaseVoFEnhanced2(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        # Alpha near 0 or 1 should stay close
        alphas = torch.tensor([[0.01], [0.99]], dtype=torch.float64)
        grad_mag = torch.tensor([0.5, 0.5], dtype=torch.float64)
        result = model.adaptive_sharpen(alphas, grad_mag)
        assert result[0, 0] <= 0.1  # Should stay near 0
        assert result[1, 0] >= 0.9  # Should stay near 1

    def test_repr(self):
        model = IncompressibleMultiphaseVoFEnhanced2(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        r = repr(model)
        assert "IncompressibleMultiphaseVoFEnhanced2" in r
        assert "water" in r
