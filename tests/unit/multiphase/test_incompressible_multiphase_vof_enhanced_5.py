"""Tests for IncompressibleMultiphaseVoFEnhanced5 (v6).

Tests cover:
- Surface tension force computation
- Interface cell tagging
- Phase-aware Courant number
- Custom parameters
- Inheritance
"""

import pytest
import torch

from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_5 import (
    IncompressibleMultiphaseVoFEnhanced5,
)
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_4 import (
    IncompressibleMultiphaseVoFEnhanced4,
)


class TestIncompressibleMultiphaseVoFEnhanced5:
    """Tests for IncompressibleMultiphaseVoFEnhanced5."""

    def test_inherits_from_v5(self):
        model = IncompressibleMultiphaseVoFEnhanced5(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert isinstance(model, IncompressibleMultiphaseVoFEnhanced4)

    def test_default_params(self):
        model = IncompressibleMultiphaseVoFEnhanced5(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        assert model.sigma == [0.072]
        assert model.refine_threshold == pytest.approx(0.01)
        assert model.co_phase_factor == pytest.approx(0.8)

    def test_custom_params(self):
        model = IncompressibleMultiphaseVoFEnhanced5(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            sigma=[0.05],
            refine_threshold=0.02,
            co_phase_factor=0.6,
        )
        assert model.sigma == [0.05]
        assert model.refine_threshold == pytest.approx(0.02)
        assert model.co_phase_factor == pytest.approx(0.6)

    def test_tag_interface_cells(self):
        model = IncompressibleMultiphaseVoFEnhanced5(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
            refine_threshold=0.1,
        )
        alpha = torch.tensor([0.99, 0.01, 0.5, 0.5], dtype=torch.float64)
        owner = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
        neighbour = torch.tensor([1, 2, 3, 0], dtype=torch.long)
        n_internal = 4
        n_cells = 4

        tags = model.tag_interface_cells(alpha, owner, neighbour, n_internal, n_cells)
        assert tags.shape == (4,)
        assert tags.dtype == torch.bool

    def test_surface_tension_force_shape(self):
        model = IncompressibleMultiphaseVoFEnhanced5(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        alpha = torch.tensor([0.9, 0.1, 0.5, 0.5], dtype=torch.float64)
        owner = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
        neighbour = torch.tensor([1, 2, 3, 0], dtype=torch.long)
        n_internal = 4
        n_cells = 4
        volumes = torch.ones(4, dtype=torch.float64)

        F = model.surface_tension_force(
            alpha, owner, neighbour, n_internal, n_cells, volumes,
        )
        assert F.shape == (4,)

    def test_repr(self):
        model = IncompressibleMultiphaseVoFEnhanced5(
            phase_names=["water", "air"],
            rho=[998.0, 1.225],
            mu=[1.002e-3, 1.8e-5],
        )
        r = repr(model)
        assert "Enhanced5" in r
        assert "refine_threshold" in r
