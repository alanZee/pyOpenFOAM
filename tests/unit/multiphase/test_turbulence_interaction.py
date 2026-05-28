"""Tests for TurbulenceInteractionModel and StandardInteraction."""

import pytest
import torch

from pyfoam.multiphase.turbulence_interaction import (
    TurbulenceInteractionModel,
    StandardInteraction,
)


class TestStandardInteraction:
    """Test the StandardInteraction model."""

    def test_registration(self):
        """standardInteraction is registered."""
        assert "standardInteraction" in TurbulenceInteractionModel.available_types()

    def test_factory_creation(self):
        """Model can be created via factory."""
        model = TurbulenceInteractionModel.create("standardInteraction")
        assert isinstance(model, StandardInteraction)

    def test_default_params(self):
        """Default parameters: C_ti=1.0, sigma_k=0.75."""
        model = StandardInteraction()
        assert model.C_ti == 1.0
        assert model.sigma_k == 0.75

    def test_custom_params(self):
        """Custom parameters are stored correctly."""
        model = StandardInteraction(C_ti=2.0, sigma_k=0.5)
        assert model.C_ti == 2.0
        assert model.sigma_k == 0.5

    def test_k_source_shape(self):
        """TKE source has correct shape."""
        model = StandardInteraction()
        n = 10
        alpha_d = torch.rand(n, dtype=torch.float64) * 0.3
        k_c = torch.ones(n, dtype=torch.float64) * 5.0
        k_d = torch.ones(n, dtype=torch.float64) * 2.0
        U_slip = torch.ones(n, dtype=torch.float64) * 1.0
        K_drag = torch.ones(n, dtype=torch.float64) * 100.0

        S_k = model.compute_k_source(alpha_d, k_c, k_d, U_slip, K_drag)
        assert S_k.shape == (n,)

    def test_k_source_non_negative(self):
        """TKE source is non-negative (energy transfer to continuous phase)."""
        model = StandardInteraction()
        n = 20
        alpha_d = torch.rand(n, dtype=torch.float64) * 0.5
        k_c = torch.rand(n, dtype=torch.float64) * 10.0 + 0.1
        k_d = torch.rand(n, dtype=torch.float64) * 5.0 + 0.1
        U_slip = torch.rand(n, dtype=torch.float64) * 3.0
        K_drag = torch.rand(n, dtype=torch.float64) * 200.0 + 1.0

        S_k = model.compute_k_source(alpha_d, k_c, k_d, U_slip, K_drag)
        assert (S_k >= 0).all()

    def test_k_source_zero_alpha(self):
        """Zero dispersed fraction yields zero source."""
        model = StandardInteraction()
        alpha_d = torch.zeros(4, dtype=torch.float64)
        k_c = torch.ones(4, dtype=torch.float64) * 5.0
        k_d = torch.ones(4, dtype=torch.float64) * 2.0
        U_slip = torch.ones(4, dtype=torch.float64) * 1.0
        K_drag = torch.ones(4, dtype=torch.float64) * 100.0

        S_k = model.compute_k_source(alpha_d, k_c, k_d, U_slip, K_drag)
        assert torch.allclose(S_k, torch.zeros(4, dtype=torch.float64), atol=1e-10)

    def test_k_source_zero_slip(self):
        """Zero slip velocity yields zero source."""
        model = StandardInteraction()
        alpha_d = torch.tensor([0.3], dtype=torch.float64)
        k_c = torch.tensor([5.0], dtype=torch.float64)
        k_d = torch.tensor([2.0], dtype=torch.float64)
        U_slip = torch.tensor([0.0], dtype=torch.float64)
        K_drag = torch.tensor([100.0], dtype=torch.float64)

        S_k = model.compute_k_source(alpha_d, k_c, k_d, U_slip, K_drag)
        assert torch.allclose(S_k, torch.tensor([0.0], dtype=torch.float64), atol=1e-10)

    def test_k_source_scales_with_drag(self):
        """TKE source scales linearly with drag coefficient."""
        model = StandardInteraction()
        alpha_d = torch.tensor([0.2], dtype=torch.float64)
        k_c = torch.tensor([5.0], dtype=torch.float64)
        k_d = torch.tensor([2.0], dtype=torch.float64)
        U_slip = torch.tensor([1.0], dtype=torch.float64)

        K1 = torch.tensor([50.0], dtype=torch.float64)
        K2 = torch.tensor([100.0], dtype=torch.float64)

        S1 = model.compute_k_source(alpha_d, k_c, k_d, U_slip, K1)
        S2 = model.compute_k_source(alpha_d, k_c, k_d, U_slip, K2)
        assert torch.allclose(S2 / S1, torch.tensor([2.0], dtype=torch.float64), rtol=1e-6)

    def test_k_source_scales_with_slip_squared(self):
        """TKE source scales with U_slip^2."""
        model = StandardInteraction()
        alpha_d = torch.tensor([0.2], dtype=torch.float64)
        k_c = torch.tensor([5.0], dtype=torch.float64)
        k_d = torch.tensor([2.0], dtype=torch.float64)
        K_drag = torch.tensor([100.0], dtype=torch.float64)

        U1 = torch.tensor([1.0], dtype=torch.float64)
        U2 = torch.tensor([3.0], dtype=torch.float64)

        S1 = model.compute_k_source(alpha_d, k_c, k_d, U1, K_drag)
        S2 = model.compute_k_source(alpha_d, k_c, k_d, U2, K_drag)
        # S2/S1 = (3/1)^2 = 9
        assert torch.allclose(S2 / S1, torch.tensor([9.0], dtype=torch.float64), rtol=1e-6)

    def test_k_source_formula(self):
        """Verify exact formula: S_k = C_ti * K_drag * U_slip^2 * alpha_d * alpha_c."""
        model = StandardInteraction(C_ti=1.5)
        alpha_d = torch.tensor([0.3], dtype=torch.float64)
        k_c = torch.tensor([5.0], dtype=torch.float64)
        k_d = torch.tensor([2.0], dtype=torch.float64)
        U_slip = torch.tensor([2.0], dtype=torch.float64)
        K_drag = torch.tensor([100.0], dtype=torch.float64)

        S_k = model.compute_k_source(alpha_d, k_c, k_d, U_slip, K_drag)
        expected = 1.5 * 100.0 * 4.0 * 0.3 * 0.7
        assert torch.allclose(S_k, torch.tensor([expected], dtype=torch.float64), atol=1e-10)

    def test_epsilon_source_shape(self):
        """Dissipation source has correct shape."""
        model = StandardInteraction()
        n = 8
        alpha_d = torch.rand(n, dtype=torch.float64) * 0.3
        k_c = torch.ones(n, dtype=torch.float64) * 5.0
        k_d = torch.ones(n, dtype=torch.float64) * 2.0
        epsilon_c = torch.ones(n, dtype=torch.float64) * 50.0
        U_slip = torch.ones(n, dtype=torch.float64) * 1.0
        K_drag = torch.ones(n, dtype=torch.float64) * 100.0

        S_eps = model.compute_epsilon_source(alpha_d, epsilon_c, k_c, k_d, U_slip, K_drag)
        assert S_eps.shape == (n,)
        assert (S_eps >= 0).all()

    def test_epsilon_source_scales_with_k(self):
        """Dissipation source is proportional to S_k * epsilon/k."""
        model = StandardInteraction(C_ti=1.0)
        alpha_d = torch.tensor([0.2], dtype=torch.float64)
        k_c = torch.tensor([10.0], dtype=torch.float64)
        k_d = torch.tensor([5.0], dtype=torch.float64)
        epsilon_c = torch.tensor([100.0], dtype=torch.float64)
        U_slip = torch.tensor([1.0], dtype=torch.float64)
        K_drag = torch.tensor([50.0], dtype=torch.float64)

        S_eps = model.compute_epsilon_source(alpha_d, epsilon_c, k_c, k_d, U_slip, K_drag)
        S_k = model.compute_k_source(alpha_d, k_c, k_d, U_slip, K_drag)
        expected = 1.0 * S_k * 100.0 / 10.0
        assert torch.allclose(S_eps, expected, atol=1e-10)

    def test_factory_unknown_raises(self):
        """Unknown model name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown turbulence interaction"):
            TurbulenceInteractionModel.create("nonexistent")
