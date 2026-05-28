"""Tests for turbulence damping models near interfaces."""

import pytest
import torch

from pyfoam.multiphase.turbulence_damping import (
    TurbulenceDampingModel,
    InterfaceDamping,
)


class TestInterfaceDamping:
    """Test the InterfaceDamping model."""

    def test_registration(self):
        """interfaceDamping is registered in the RTS registry."""
        assert "interfaceDamping" in TurbulenceDampingModel.available_types()

    def test_factory_creation(self):
        """Model can be created via the factory method."""
        model = TurbulenceDampingModel.create("interfaceDamping")
        assert isinstance(model, InterfaceDamping)

    def test_default_params(self):
        """Default parameters: damping_coeff=10, alpha_min=0.01, alpha_max=0.99."""
        model = InterfaceDamping()
        assert model.damping_coeff == 10.0
        assert model.alpha_min == 0.01
        assert model.alpha_max == 0.99

    def test_custom_params(self):
        """Custom parameters are stored correctly."""
        model = InterfaceDamping(damping_coeff=5.0, alpha_min=0.05, alpha_max=0.95)
        assert model.damping_coeff == 5.0
        assert model.alpha_min == 0.05
        assert model.alpha_max == 0.95

    def test_damping_factor_pure_phases(self):
        """Pure phases (alpha=0 or 1) have zero damping factor."""
        model = InterfaceDamping(damping_coeff=10.0)
        alpha = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        f = model.compute_damping_factor(alpha)
        assert torch.allclose(f, torch.zeros(4, dtype=torch.float64), atol=1e-10)

    def test_damping_factor_at_interface(self):
        """Interface region (alpha=0.5) has maximum damping factor."""
        model = InterfaceDamping(damping_coeff=10.0)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        f = model.compute_damping_factor(alpha)
        # 4 * 0.5 * 0.5 * 10 = 10
        expected = 10.0
        assert torch.allclose(f, torch.tensor([expected], dtype=torch.float64), atol=1e-10)

    def test_damping_factor_outside_threshold(self):
        """Alpha outside [alpha_min, alpha_max] has zero damping."""
        model = InterfaceDamping(damping_coeff=10.0, alpha_min=0.1, alpha_max=0.9)
        alpha = torch.tensor([0.001, 0.999], dtype=torch.float64)
        f = model.compute_damping_factor(alpha)
        assert torch.allclose(f, torch.zeros(2, dtype=torch.float64), atol=1e-10)

    def test_damping_factor_symmetric(self):
        """Damping factor is symmetric around alpha=0.5."""
        model = InterfaceDamping(damping_coeff=10.0)
        alpha = torch.tensor([0.3, 0.7], dtype=torch.float64)
        f = model.compute_damping_factor(alpha)
        assert torch.allclose(f[0], f[1], atol=1e-10)

    def test_damp_k_reduces_at_interface(self):
        """k is reduced at the interface."""
        model = InterfaceDamping(damping_coeff=10.0)
        alpha = torch.tensor([0.5, 0.0, 1.0], dtype=torch.float64)
        k = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        # At interface: k * exp(-10) ~ 0.0000454
        assert k_damped[0] < 1.0
        # Pure phases: unchanged
        assert torch.allclose(k_damped[1], torch.tensor(100.0, dtype=torch.float64))
        assert torch.allclose(k_damped[2], torch.tensor(100.0, dtype=torch.float64))

    def test_damp_epsilon_reduces_at_interface(self):
        """epsilon is reduced at the interface."""
        model = InterfaceDamping(damping_coeff=10.0)
        alpha = torch.tensor([0.5, 0.0, 1.0], dtype=torch.float64)
        eps = torch.tensor([50.0, 50.0, 50.0], dtype=torch.float64)
        eps_damped = model.damp_epsilon(alpha, eps)
        assert eps_damped[0] < 1.0
        assert torch.allclose(eps_damped[1], torch.tensor(50.0, dtype=torch.float64))
        assert torch.allclose(eps_damped[2], torch.tensor(50.0, dtype=torch.float64))

    def test_damp_omega_reduces_at_interface(self):
        """omega is reduced at the interface."""
        model = InterfaceDamping(damping_coeff=10.0)
        alpha = torch.tensor([0.5, 0.0], dtype=torch.float64)
        omega = torch.tensor([200.0, 200.0], dtype=torch.float64)
        omega_damped = model.damp_omega(alpha, omega)
        assert omega_damped[0] < 1.0
        assert torch.allclose(omega_damped[1], torch.tensor(200.0, dtype=torch.float64))

    def test_zero_damping_coeff_no_effect(self):
        """damping_coeff=0 should leave fields unchanged."""
        model = InterfaceDamping(damping_coeff=0.0)
        alpha = torch.tensor([0.5, 0.3, 0.7], dtype=torch.float64)
        k = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert torch.allclose(k_damped, k, atol=1e-10)

    def test_higher_damping_coeff_more_reduction(self):
        """Higher damping coefficient produces more reduction."""
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([100.0], dtype=torch.float64)

        model_low = InterfaceDamping(damping_coeff=1.0)
        model_high = InterfaceDamping(damping_coeff=20.0)

        k_low = model_low.damp_k(alpha, k)
        k_high = model_high.damp_k(alpha, k)

        assert k_low > k_high

    def test_batch_processing(self):
        """Works correctly with batched tensors."""
        model = InterfaceDamping(damping_coeff=10.0)
        n = 100
        alpha = torch.rand(n, dtype=torch.float64)
        k = torch.ones(n, dtype=torch.float64) * 10.0
        k_damped = model.damp_k(alpha, k)
        assert k_damped.shape == (n,)
        # k_damped <= k always
        assert (k_damped <= k + 1e-10).all()
