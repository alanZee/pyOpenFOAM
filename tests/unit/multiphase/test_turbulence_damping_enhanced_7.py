"""Tests for TurbulenceDamping7EnhancedModel (v7).

Tests cover:
- ML-assisted damping
- Anisotropic tensor damping
- Shear layer damping
- Registry and factory
"""

import pytest
import torch

from pyfoam.multiphase.turbulence_damping_enhanced_7 import (
    TurbulenceDamping7EnhancedModel,
    MLAssistedDamping,
    AnisotropicTensorDamping,
    ShearLayerDamping,
)


class TestTurbulenceDamping7EnhancedModel:
    """Tests for v7 turbulence damping models."""

    def test_registry_has_models(self):
        available = TurbulenceDamping7EnhancedModel.available_types()
        assert "mlAssisted" in available
        assert "anisotropicTensor" in available
        assert "shearLayer" in available

    def test_factory_create_ml_assisted(self):
        model = TurbulenceDamping7EnhancedModel.create("mlAssisted")
        assert isinstance(model, MLAssistedDamping)

    def test_factory_create_anisotropic(self):
        model = TurbulenceDamping7EnhancedModel.create("anisotropicTensor")
        assert isinstance(model, AnisotropicTensorDamping)

    def test_factory_create_shear_layer(self):
        model = TurbulenceDamping7EnhancedModel.create("shearLayer")
        assert isinstance(model, ShearLayerDamping)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError):
            TurbulenceDamping7EnhancedModel.create("nonexistent")

    def test_ml_assisted_basic(self):
        model = MLAssistedDamping(damping_coeff=10.0)
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        result = model.compute_damping_factor(alpha)
        assert result.shape == (20,)
        assert (result >= 0).all()

    def test_ml_assisted_with_gradient(self):
        model = MLAssistedDamping(damping_coeff=10.0)
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        grad = torch.rand(20, dtype=torch.float64) * 0.5

        result = model.compute_damping_factor(alpha, grad_alpha_mag=grad)
        assert result.shape == (20,)

    def test_ml_assisted_custom_weights(self):
        weights = [1.0, 3.0, 0.5, -0.5, 0.2, -0.1]
        model = MLAssistedDamping(damping_coeff=10.0, weights=weights)
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        result = model.compute_damping_factor(alpha)
        assert result.shape == (20,)

    def test_anisotropic_tensor_shape(self):
        model = AnisotropicTensorDamping(damping_coeff=10.0)
        alpha = torch.rand(10, dtype=torch.float64) * 0.4 + 0.3
        D = model.compute_damping_tensor(alpha)
        assert D.shape == (10, 3, 3)

    def test_anisotropic_tensor_with_normal(self):
        model = AnisotropicTensorDamping(damping_coeff=10.0, anisotropy_beta=0.5)
        alpha = torch.rand(10, dtype=torch.float64) * 0.4 + 0.3
        normal = torch.randn(10, 3, dtype=torch.float64)

        D = model.compute_damping_tensor(alpha, normal=normal)
        assert D.shape == (10, 3, 3)

    def test_anisotropic_scalar_factor(self):
        model = AnisotropicTensorDamping(damping_coeff=10.0)
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        result = model.compute_damping_factor(alpha)
        assert result.shape == (20,)

    def test_shear_layer_basic(self):
        model = ShearLayerDamping(damping_coeff=10.0)
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        result = model.compute_damping_factor(alpha)
        assert result.shape == (20,)

    def test_shear_layer_with_shear_rate(self):
        model = ShearLayerDamping(damping_coeff=10.0, C_shear=0.3)
        alpha = torch.rand(20, dtype=torch.float64) * 0.4 + 0.3
        grad = torch.rand(20, dtype=torch.float64) * 0.5
        shear = torch.rand(20, dtype=torch.float64) * 5.0

        result = model.compute_damping_factor(
            alpha, grad_alpha_mag=grad, shear_rate=shear,
        )
        assert result.shape == (20,)
