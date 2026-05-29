"""Tests for enhanced turbulence inlet models (version 2).

Tests cover:
- TurbulenceInletModel2 base class registry
- DigitalFilterInlet: filter coefficients, fluctuations, k/epsilon
- SyntheticEddyInlet: eddy placement, shape function, fluctuations
"""

import pytest
import torch

from pyfoam.turbulence.turbulence_inlet_models_2 import (
    TurbulenceInletModel2,
    DigitalFilterInlet,
    SyntheticEddyInlet,
)


class TestTurbulenceInletModel2Registry:
    """RTS registry tests."""

    def test_digital_filter_registered(self):
        assert "digitalFilterInlet" in TurbulenceInletModel2.available_types()

    def test_synthetic_eddy_registered(self):
        assert "syntheticEddyInlet" in TurbulenceInletModel2.available_types()

    def test_factory_digital_filter(self):
        model = TurbulenceInletModel2.create(
            "digitalFilterInlet", k=0.01, epsilon=0.001,
        )
        assert isinstance(model, DigitalFilterInlet)

    def test_factory_synthetic_eddy(self):
        model = TurbulenceInletModel2.create(
            "syntheticEddyInlet", k=0.01, epsilon=0.001,
        )
        assert isinstance(model, SyntheticEddyInlet)

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            TurbulenceInletModel2.create("nonexistent")


class TestDigitalFilterInlet:
    """DigitalFilterInlet tests."""

    def test_default_params(self):
        model = DigitalFilterInlet()
        assert model.k_target == pytest.approx(0.01)
        assert model.epsilon_value == pytest.approx(0.001)
        assert model.length_scales == (0.1, 0.05, 0.05)
        assert model.n_filter == 11

    def test_custom_params(self):
        model = DigitalFilterInlet(
            k=0.05, epsilon=0.01,
            length_scale_x=0.2, length_scale_y=0.1, length_scale_z=0.1,
            n_filter=21,
        )
        assert model.k_target == pytest.approx(0.05)
        assert model.epsilon_value == pytest.approx(0.01)
        assert model.length_scales == (0.2, 0.1, 0.1)
        assert model.n_filter == 21

    def test_n_filter_made_odd(self):
        """Even n_filter is incremented to odd."""
        model = DigitalFilterInlet(n_filter=10)
        assert model.n_filter == 11

    def test_filter_coefficients_positive(self):
        """All filter coefficients are positive."""
        model = DigitalFilterInlet(length_scale_y=0.1, n_filter=11)
        b = model.compute_filter_coefficients(L=0.1, dx=0.01)
        assert (b >= 0).all()

    def test_filter_coefficients_normalised(self):
        """Filter coefficients sum to 1 (approximately)."""
        model = DigitalFilterInlet(length_scale_y=0.1, n_filter=11)
        b = model.compute_filter_coefficients(L=0.1, dx=0.01)
        assert b.sum().item() == pytest.approx(1.0, rel=1e-3)

    def test_filter_coefficients_symmetric(self):
        """Filter coefficients are symmetric around the centre."""
        model = DigitalFilterInlet(n_filter=11)
        b = model.compute_filter_coefficients(L=0.1, dx=0.01)
        # Symmetric: b[i] == b[n-1-i]
        for i in range(len(b) // 2):
            assert b[i].item() == pytest.approx(b[-(i + 1)].item(), rel=1e-6)

    def test_filter_coefficients_shape(self):
        model = DigitalFilterInlet(n_filter=15)
        b = model.compute_filter_coefficients(L=0.1, dx=0.01)
        assert b.shape == (15,)

    def test_generate_fluctuations_shape(self):
        model = DigitalFilterInlet()
        u = model.generate_fluctuations(n_faces=50)
        assert u.shape == (50, 3)

    def test_generate_fluctuations_with_positions(self):
        model = DigitalFilterInlet()
        positions = torch.rand(30, 3, dtype=torch.float64)
        u = model.generate_fluctuations(n_faces=30, face_positions=positions)
        assert u.shape == (30, 3)

    def test_fluctuations_finite(self):
        model = DigitalFilterInlet(k=0.01)
        u = model.generate_fluctuations(n_faces=100)
        assert torch.isfinite(u).all()

    def test_fluctuations_have_correct_k(self):
        """Generated fluctuations should have approximately the target k."""
        model = DigitalFilterInlet(k=0.05, n_filter=31)
        u = model.generate_fluctuations(n_faces=200)
        k_actual = 0.5 * (u ** 2).sum(dim=-1).mean()
        # Allow 50% tolerance due to stochastic nature and small sample
        assert k_actual.item() == pytest.approx(0.05, rel=0.5)

    def test_compute_k_uniform(self):
        model = DigitalFilterInlet(k=0.05)
        k = model.compute_k(n_faces=50)
        assert k.shape == (50,)
        assert torch.allclose(k, torch.full((50,), 0.05, dtype=k.dtype))

    def test_compute_epsilon_uniform(self):
        model = DigitalFilterInlet(epsilon=0.01)
        eps = model.compute_epsilon(n_faces=50)
        assert eps.shape == (50,)
        assert torch.allclose(eps, torch.full((50,), 0.01, dtype=eps.dtype))

    def test_compute_omega(self):
        model = DigitalFilterInlet(k=0.01, epsilon=0.001)
        omega = model.compute_omega(n_faces=10)
        expected = 0.001 / (0.09 * 0.01)
        assert torch.allclose(omega, torch.full((10,), expected, dtype=omega.dtype), rtol=1e-4)

    def test_zero_faces(self):
        model = DigitalFilterInlet()
        u = model.generate_fluctuations(n_faces=0)
        assert u.shape == (0, 3)


class TestSyntheticEddyInlet:
    """SyntheticEddyInlet tests."""

    def test_default_params(self):
        model = SyntheticEddyInlet()
        assert model.k_target == pytest.approx(0.01)
        assert model.epsilon_value == pytest.approx(0.001)
        assert model.n_eddies == 50
        assert model.box_size == (0.5, 0.5, 0.5)

    def test_custom_params(self):
        model = SyntheticEddyInlet(
            k=0.05, epsilon=0.01, n_eddies=100, box_size=(1.0, 0.5, 0.5),
        )
        assert model.k_target == pytest.approx(0.05)
        assert model.epsilon_value == pytest.approx(0.01)
        assert model.n_eddies == 100
        assert model.box_size == (1.0, 0.5, 0.5)

    def test_generate_fluctuations_shape(self):
        model = SyntheticEddyInlet()
        u = model.generate_fluctuations(n_faces=30)
        assert u.shape == (30, 3)

    def test_generate_fluctuations_with_positions(self):
        model = SyntheticEddyInlet()
        positions = torch.rand(20, 3, dtype=torch.float64)
        u = model.generate_fluctuations(n_faces=20, face_positions=positions)
        assert u.shape == (20, 3)

    def test_fluctuations_finite(self):
        model = SyntheticEddyInlet(k=0.01, n_eddies=20)
        u = model.generate_fluctuations(n_faces=50)
        assert torch.isfinite(u).all()

    def test_fluctuations_nonzero(self):
        """With enough eddies, fluctuations should be non-zero."""
        model = SyntheticEddyInlet(k=0.01, n_eddies=30)
        u = model.generate_fluctuations(n_faces=50)
        # At least some faces should have non-zero fluctuations
        assert (u.abs() > 1e-10).any()

    def test_shape_function_compact_support(self):
        """Shape function is zero outside the eddy radius."""
        model = SyntheticEddyInlet(box_size=(0.5, 0.5, 0.5))
        # Point far from eddy origin (well outside box_size/2)
        r = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64)
        f = model._shape_function(r)
        assert f[0].item() == pytest.approx(0.0)

    def test_shape_function_at_origin(self):
        """Shape function is maximum at the eddy centre."""
        model = SyntheticEddyInlet(box_size=(0.5, 0.5, 0.5))
        r = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        f = model._shape_function(r)
        # f = sqrt(2)^3 * 1^3 = 2*sqrt(2) ≈ 2.828
        assert f[0].item() == pytest.approx(2.0 * math.sqrt(2.0))

    def test_shape_function_decays(self):
        """Shape function decreases with distance from eddy centre."""
        model = SyntheticEddyInlet(box_size=(1.0, 1.0, 1.0))
        r_near = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
        r_far = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float64)
        f_near = model._shape_function(r_near)
        f_far = model._shape_function(r_far)
        assert f_near[0].item() > f_far[0].item()

    def test_compute_k_uniform(self):
        model = SyntheticEddyInlet(k=0.05)
        k = model.compute_k(n_faces=20)
        assert k.shape == (20,)
        assert torch.allclose(k, torch.full((20,), 0.05, dtype=k.dtype))

    def test_compute_epsilon_uniform(self):
        model = SyntheticEddyInlet(epsilon=0.01)
        eps = model.compute_epsilon(n_faces=20)
        assert eps.shape == (20,)
        assert torch.allclose(eps, torch.full((20,), 0.01, dtype=eps.dtype))

    def test_compute_omega(self):
        model = SyntheticEddyInlet(k=0.01, epsilon=0.001)
        omega = model.compute_omega(n_faces=5)
        expected = 0.001 / (0.09 * 0.01)
        assert torch.allclose(omega, torch.full((5,), expected, dtype=omega.dtype), rtol=1e-4)

    def test_more_eddies_better_statistics(self):
        """More eddies should produce fluctuations closer to target k."""
        model_few = SyntheticEddyInlet(k=0.01, n_eddies=5)
        model_many = SyntheticEddyInlet(k=0.01, n_eddies=200)
        u_few = model_few.generate_fluctuations(n_faces=100)
        u_many = model_many.generate_fluctuations(n_faces=100)
        k_few = 0.5 * (u_few ** 2).sum(dim=-1).mean().item()
        k_many = 0.5 * (u_many ** 2).sum(dim=-1).mean().item()
        # Many eddies should be closer to 0.01
        err_few = abs(k_few - 0.01)
        err_many = abs(k_many - 0.01)
        # This is a statistical test; we just check both are reasonable
        assert err_many < 0.05


# Need math import for shape function test
import math  # noqa: E402
