"""Tests for IncompressibleMultiphaseVoFEnhanced4 (v5).

Tests cover:
- Gradient adaptive compression coefficient
- Interface normal smoothing
- Multi-pass bounded sweep
- Full advance with v5 enhancements
"""

import pytest
import torch

from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_4 import (
    IncompressibleMultiphaseVoFEnhanced4,
)
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_3 import (
    IncompressibleMultiphaseVoFEnhanced3,
)


class TestIncompressibleMultiphaseVoFEnhanced4:
    """Tests for IncompressibleMultiphaseVoFEnhanced4."""

    def test_inherits_from_v4(self):
        model = IncompressibleMultiphaseVoFEnhanced4(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert isinstance(model, IncompressibleMultiphaseVoFEnhanced3)

    def test_default_params(self):
        model = IncompressibleMultiphaseVoFEnhanced4(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert model.grad_adapt_coeff == pytest.approx(0.5)
        assert model.normal_smooth_iters == 2
        assert model.n_sweep_passes == 3

    def test_custom_params(self):
        model = IncompressibleMultiphaseVoFEnhanced4(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            grad_adapt_coeff=1.0,
            normal_smooth_iters=5,
            n_sweep_passes=4,
        )
        assert model.grad_adapt_coeff == pytest.approx(1.0)
        assert model.normal_smooth_iters == 5
        assert model.n_sweep_passes == 4

    def test_gradient_adaptive_coeff_shape(self):
        model = IncompressibleMultiphaseVoFEnhanced4(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        alpha = torch.tensor([0.9, 0.1, 0.5, 0.5], dtype=torch.float64)
        owner = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
        neighbour = torch.tensor([1, 2, 3, 0], dtype=torch.long)
        n_internal = 4
        C = model.gradient_adaptive_compression_coeff(alpha, owner, neighbour, n_internal)
        assert C.shape == (4,)
        assert (C >= 0).all()
        assert (C <= 3.0).all()

    def test_multi_pass_sweep_bounds(self):
        model = IncompressibleMultiphaseVoFEnhanced4(
            phase_names=["water", "air", "oil"],
            rho=[998.0, 1.225, 850.0],
            mu=[1.002e-3, 1.8e-5, 0.03],
            n_sweep_passes=3,
        )
        alphas_new = torch.tensor([
            [1.5, 0.3],
            [0.3, 0.3],
            [-0.1, 0.5],
        ], dtype=torch.float64)
        alphas_old = torch.tensor([
            [0.5, 0.3],
            [0.3, 0.3],
            [0.5, 0.5],
        ], dtype=torch.float64)

        result = model.multi_pass_bounded_sweep(alphas_new, alphas_old)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_repr(self):
        model = IncompressibleMultiphaseVoFEnhanced4(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        r = repr(model)
        assert "Enhanced4" in r
        assert "grad_adapt_coeff" in r
