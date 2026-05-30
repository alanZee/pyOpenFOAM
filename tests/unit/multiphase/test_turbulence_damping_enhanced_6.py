"""Tests for TurbulenceDamping6EnhancedModel (v6).

Tests cover:
- LES-aware damping
- Dynamic coefficient damping
- Topology-aware damping
- Registry and factory
"""

import pytest
import torch

from pyfoam.multiphase.turbulence_damping_enhanced_6 import (
    TurbulenceDamping6EnhancedModel,
    LESAwareDamping,
    DynamicCoefficientDamping,
    TopologyAwareDamping,
)


class TestTurbulenceDamping6EnhancedModel:
    """Tests for v6 turbulence damping models."""

    def test_registry_has_models(self):
        available = TurbulenceDamping6EnhancedModel.available_types()
        assert "lesAware" in available
        assert "dynamicCoefficient" in available
        assert "topologyAware" in available

    def test_factory_create_les_aware(self):
        model = TurbulenceDamping6EnhancedModel.create("lesAware")
        assert isinstance(model, LESAwareDamping)

    def test_factory_create_dynamic_coefficient(self):
        model = TurbulenceDamping6EnhancedModel.create("dynamicCoefficient")
        assert isinstance(model, DynamicCoefficientDamping)

    def test_factory_create_topology_aware(self):
        model = TurbulenceDamping6EnhancedModel.create("topologyAware")
        assert isinstance(model, TopologyAwareDamping)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError):
            TurbulenceDamping6EnhancedModel.create("nonexistent")

    def test_les_aware_basic(self):
        model = LESAwareDamping(damping_coeff=10.0, delta_interface=1e-3)
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        result = model.compute_damping_factor(alpha)
        assert result.shape == (20,)
        assert (result >= 0).all()

    def test_les_aware_with_grid_delta(self):
        model = LESAwareDamping(damping_coeff=10.0, sgs_weight=0.5)
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        delta = torch.rand(20, dtype=torch.float64) * 0.01
        sgs_energy = torch.rand(20, dtype=torch.float64) * 0.1

        result = model.compute_damping_factor(
            alpha, delta=delta, sgs_energy=sgs_energy,
        )
        assert result.shape == (20,)

    def test_dynamic_coefficient_basic(self):
        model = DynamicCoefficientDamping(damping_coeff=10.0)
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        result = model.compute_damping_factor(alpha)
        assert result.shape == (20,)

    def test_dynamic_coefficient_with_k_epsilon(self):
        model = DynamicCoefficientDamping(damping_coeff=10.0, Re_t_ref=100.0)
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        k = torch.rand(20, dtype=torch.float64) * 0.1
        epsilon = torch.rand(20, dtype=torch.float64) * 0.01
        nu = 1e-6

        result = model.compute_damping_factor(
            alpha, k=k, epsilon=epsilon, nu=nu,
        )
        assert result.shape == (20,)

    def test_topology_aware_basic(self):
        model = TopologyAwareDamping(damping_coeff=10.0)
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        result = model.compute_damping_factor(alpha)
        assert result.shape == (20,)

    def test_topology_aware_with_grad(self):
        model = TopologyAwareDamping(
            damping_coeff=10.0,
            continuous_weight=1.0,
            dispersed_weight=2.0,
        )
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        grad_alpha = torch.rand(20, dtype=torch.float64) * 0.5
        a_i = torch.rand(20, dtype=torch.float64) * 100

        result = model.compute_damping_factor(
            alpha, grad_alpha_mag=grad_alpha, a_i=a_i, d32=3e-3,
        )
        assert result.shape == (20,)
