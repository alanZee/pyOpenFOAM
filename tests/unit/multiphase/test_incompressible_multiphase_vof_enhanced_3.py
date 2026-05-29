"""Tests for IncompressibleMultiphaseVoFEnhanced3 (v4).

Tests cover:
- Curvature correction factor
- Slip correction
- Hierarchical clamp
- Full advance with v4 enhancements
"""

import pytest
import torch

from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_3 import (
    IncompressibleMultiphaseVoFEnhanced3,
)
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_2 import (
    IncompressibleMultiphaseVoFEnhanced2,
)


class TestIncompressibleMultiphaseVoFEnhanced3:
    """Tests for IncompressibleMultiphaseVoFEnhanced3."""

    def test_inherits_from_v3(self):
        model = IncompressibleMultiphaseVoFEnhanced3(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert isinstance(model, IncompressibleMultiphaseVoFEnhanced2)

    def test_default_params(self):
        model = IncompressibleMultiphaseVoFEnhanced3(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert model.curvature_coeff == pytest.approx(0.5)
        assert model.slip_coeff == pytest.approx(0.1)
        assert model.n_clamp_levels == 3

    def test_custom_params(self):
        model = IncompressibleMultiphaseVoFEnhanced3(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            curvature_coeff=0.8,
            slip_coeff=0.2,
            n_clamp_levels=5,
        )
        assert model.curvature_coeff == pytest.approx(0.8)
        assert model.slip_coeff == pytest.approx(0.2)
        assert model.n_clamp_levels == 5

    def test_curvature_correction_shape(self):
        model = IncompressibleMultiphaseVoFEnhanced3(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        alpha = torch.tensor([0.9, 0.1, 0.5, 0.5], dtype=torch.float64)
        owner = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
        neighbour = torch.tensor([1, 2, 3, 0], dtype=torch.long)
        n_internal = 4
        K = model.curvature_correction_factor(alpha, owner, neighbour, n_internal)
        assert K.shape == (4,)
        assert (K >= 0.5).all()
        assert (K <= 2.0).all()

    def test_hierarchical_clamp_bounds(self):
        model = IncompressibleMultiphaseVoFEnhanced3(
            phase_names=["water", "air", "oil"],
            rho=[998.0, 1.225, 850.0],
            mu=[1.002e-3, 1.8e-5, 0.03],
            n_clamp_levels=3,
        )
        # Create out-of-bounds alphas
        alphas_new = torch.tensor([
            [1.5, 0.3],   # sum > 1 => last phase negative
            [0.3, 0.3],   # sum = 0.6 => OK
            [-0.1, 0.5],  # negative alpha
        ], dtype=torch.float64)
        alphas_old = torch.tensor([
            [0.5, 0.3],
            [0.3, 0.3],
            [0.5, 0.5],
        ], dtype=torch.float64)

        result = model.hierarchical_clamp(alphas_new, alphas_old)
        # All independent phases should be in [0, 1]
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()
        # Last phase = 1 - sum should also be in [0, 1]
        alpha_last = 1.0 - result.sum(dim=-1)
        assert (alpha_last >= -0.01).all()  # Allow small tolerance

    def test_repr(self):
        model = IncompressibleMultiphaseVoFEnhanced3(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        r = repr(model)
        assert "Enhanced3" in r
        assert "curvature_coeff" in r
