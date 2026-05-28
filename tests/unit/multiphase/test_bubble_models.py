"""Tests for bubble dynamics models.

Tests cover:
- BubbleModel RTS registry
- ConstantBubble: constant diameter model
- BubbleBreakup: breakup/coalescence equilibrium model
"""

import pytest
import torch

from pyfoam.multiphase.bubble_models import (
    BubbleModel,
    ConstantBubble,
    BubbleBreakup,
)


class TestBubbleModelRegistry:
    """BubbleModel RTS registration tests."""

    def test_constant_bubble_registered(self):
        assert "constantBubble" in BubbleModel.available_types()

    def test_bubble_breakup_registered(self):
        assert "bubbleBreakup" in BubbleModel.available_types()

    def test_factory_create_constant(self):
        model = BubbleModel.create("constantBubble", d=0.005)
        assert isinstance(model, ConstantBubble)
        assert model.d == pytest.approx(0.005)

    def test_factory_create_breakup(self):
        model = BubbleModel.create("bubbleBreakup", d_min=1e-5, d_max=0.02)
        assert isinstance(model, BubbleBreakup)
        assert model.d_min == pytest.approx(1e-5)
        assert model.d_max == pytest.approx(0.02)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown BubbleModel"):
            BubbleModel.create("nonexistentModel")

    def test_available_types_sorted(self):
        types = BubbleModel.available_types()
        assert types == sorted(types)

    def test_duplicate_registration_raises(self):
        """Registering the same name twice raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            @BubbleModel.register("constantBubble")
            class _Duplicate:
                pass


class TestConstantBubble:
    """Constant bubble diameter model tests."""

    def test_default_diameter(self):
        model = ConstantBubble()
        assert model.d == pytest.approx(0.003)

    def test_custom_diameter(self):
        model = ConstantBubble(d=0.005)
        assert model.d == pytest.approx(0.005)

    def test_compute_diameter_shape(self):
        model = ConstantBubble(d=0.002)
        alpha = torch.zeros(10, dtype=torch.float64)
        d = model.compute_diameter(alpha, n_cells=10)
        assert d.shape == (10,)

    def test_compute_diameter_constant(self):
        model = ConstantBubble(d=0.004)
        alpha = torch.rand(5, dtype=torch.float64)
        d = model.compute_diameter(alpha, n_cells=5)
        assert torch.allclose(d, torch.full((5,), 0.004, dtype=torch.float64))

    def test_compute_diameter_independent_of_alpha(self):
        """Constant bubble diameter should not depend on volume fraction."""
        model = ConstantBubble(d=0.003)
        alpha_zero = torch.zeros(10, dtype=torch.float64)
        alpha_high = torch.full((10,), 0.5, dtype=torch.float64)
        d_zero = model.compute_diameter(alpha_zero, n_cells=10)
        d_high = model.compute_diameter(alpha_high, n_cells=10)
        assert torch.allclose(d_zero, d_high)


class TestBubbleBreakup:
    """Bubble breakup/coalescence model tests."""

    def test_default_parameters(self):
        model = BubbleBreakup()
        assert model.d_min == pytest.approx(1e-4)
        assert model.d_max == pytest.approx(0.01)
        assert model.C_breakup == pytest.approx(0.2)
        assert model.We_crit == pytest.approx(1.0)

    def test_custom_parameters(self):
        model = BubbleBreakup(d_min=1e-5, d_max=0.02, C_breakup=0.3, We_crit=2.0)
        assert model.d_min == pytest.approx(1e-5)
        assert model.d_max == pytest.approx(0.02)
        assert model.C_breakup == pytest.approx(0.3)
        assert model.We_crit == pytest.approx(2.0)

    def test_compute_diameter_shape(self):
        model = BubbleBreakup()
        alpha = torch.zeros(10, dtype=torch.float64)
        d = model.compute_diameter(alpha, n_cells=10)
        assert d.shape == (10,)

    def test_zero_alpha_near_dmax(self):
        """At zero volume fraction, diameter should be close to d_max."""
        model = BubbleBreakup(d_min=1e-4, d_max=0.01, C_breakup=0.2)
        alpha = torch.zeros(5, dtype=torch.float64)
        d = model.compute_diameter(alpha, n_cells=5)
        # d_eq = d_min + (d_max - d_min) * (1 - 0) = d_max
        expected = torch.full((5,), 0.01, dtype=torch.float64)
        assert torch.allclose(d, expected)

    def test_high_alpha_smaller_diameter(self):
        """Higher volume fraction should give smaller diameter."""
        model = BubbleBreakup(d_min=1e-4, d_max=0.01, C_breakup=0.2)
        alpha_low = torch.full((5,), 0.01, dtype=torch.float64)
        alpha_high = torch.full((5,), 0.5, dtype=torch.float64)
        d_low = model.compute_diameter(alpha_low, n_cells=5)
        d_high = model.compute_diameter(alpha_high, n_cells=5)
        assert (d_low > d_high).all()

    def test_diameter_clamped_to_range(self):
        """Diameter should always be within [d_min, d_max]."""
        model = BubbleBreakup(d_min=1e-4, d_max=0.01, C_breakup=0.2)
        alpha = torch.rand(20, dtype=torch.float64)
        d = model.compute_diameter(alpha, n_cells=20)
        assert (d >= model.d_min - 1e-10).all()
        assert (d <= model.d_max + 1e-10).all()

    def test_with_epsilon_reduces_diameter(self):
        """High dissipation rate should reduce bubble diameter."""
        model = BubbleBreakup(d_min=1e-4, d_max=0.01, We_crit=1.0)
        alpha = torch.full((5,), 0.1, dtype=torch.float64)
        eps_low = torch.full((5,), 0.001, dtype=torch.float64)
        eps_high = torch.full((5,), 100.0, dtype=torch.float64)
        d_low_eps = model.compute_diameter(alpha, n_cells=5, epsilon=eps_low)
        d_high_eps = model.compute_diameter(alpha, n_cells=5, epsilon=eps_high)
        # Higher epsilon → more breakup → smaller diameter
        assert (d_low_eps >= d_high_eps).all()

    def test_diameter_positive(self):
        """Diameter should always be positive."""
        model = BubbleBreakup()
        alpha = torch.rand(50, dtype=torch.float64)
        d = model.compute_diameter(alpha, n_cells=50)
        assert (d > 0).all()

    def test_monotone_in_alpha(self):
        """Diameter should monotonically decrease with increasing alpha."""
        model = BubbleBreakup(d_min=1e-4, d_max=0.01, C_breakup=0.2)
        alphas = torch.linspace(0.0, 0.8, 20, dtype=torch.float64)
        d = model.compute_diameter(alphas, n_cells=20)
        # Each successive diameter should be <= previous
        for i in range(1, len(d)):
            assert d[i] <= d[i - 1] + 1e-10
