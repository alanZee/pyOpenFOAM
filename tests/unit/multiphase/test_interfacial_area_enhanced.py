"""Tests for enhanced interfacial area models.

Tests cover:
- SauterMeanInterfacialArea: dense correction and RTS registration
- BreakupCoalescenceInterfacialArea: equilibrium and deviation
- BlendedInterfacialArea: sigmoid blending between two models
"""

import pytest
import torch

from pyfoam.multiphase.interfacial_area import InterfacialAreaModel
from pyfoam.multiphase.interfacial_area_enhanced import (
    SauterMeanInterfacialArea,
    BreakupCoalescenceInterfacialArea,
    BlendedInterfacialArea,
)


class TestSauterMeanInterfacialArea:
    """Tests for SauterMeanInterfacialArea."""

    def test_registered(self):
        assert "sauterMean" in InterfacialAreaModel.available_types()

    def test_factory_create(self):
        model = InterfacialAreaModel.create("sauterMean", d32_0=2e-3)
        assert isinstance(model, SauterMeanInterfacialArea)
        assert model.d32_0 == pytest.approx(2e-3)

    def test_default_params(self):
        model = SauterMeanInterfacialArea()
        assert model.d32_0 == pytest.approx(3e-3)
        assert model.alpha_min == pytest.approx(1e-4)
        assert model.richardson_zaki_n == pytest.approx(2.0)

    def test_custom_params(self):
        model = SauterMeanInterfacialArea(d32_0=1e-3, alpha_min=1e-5, richardson_zaki_n=3.0)
        assert model.d32_0 == pytest.approx(1e-3)
        assert model.alpha_min == pytest.approx(1e-5)
        assert model.richardson_zaki_n == pytest.approx(3.0)

    def test_zero_alpha_gives_zero_area(self):
        model = SauterMeanInterfacialArea()
        alpha = torch.zeros(5, dtype=torch.float64)
        a = model.compute(alpha, n_cells=5)
        assert torch.allclose(a, torch.zeros(5, dtype=torch.float64))

    def test_below_alpha_min_gives_zero(self):
        model = SauterMeanInterfacialArea(alpha_min=1e-3)
        alpha = torch.full((5,), 1e-5, dtype=torch.float64)
        a = model.compute(alpha, n_cells=5)
        assert torch.allclose(a, torch.zeros(5, dtype=torch.float64))

    def test_formula_with_n_equals_one(self):
        """With n=1, should match simple 6*alpha/d32."""
        model = SauterMeanInterfacialArea(d32_0=1e-3, alpha_min=1e-10, richardson_zaki_n=1.0)
        alpha = torch.tensor([0.1, 0.3], dtype=torch.float64)
        a = model.compute(alpha, n_cells=2)
        expected = 6.0 * alpha / 1e-3
        assert torch.allclose(a, expected, atol=1e-3)

    def test_dense_correction_reduces_area(self):
        """Higher n should reduce area at same alpha."""
        alpha = torch.full((5,), 0.5, dtype=torch.float64)
        model_low = SauterMeanInterfacialArea(d32_0=1e-3, alpha_min=1e-10, richardson_zaki_n=1.0)
        model_high = SauterMeanInterfacialArea(d32_0=1e-3, alpha_min=1e-10, richardson_zaki_n=3.0)
        a_low = model_low.compute(alpha, n_cells=5)
        a_high = model_high.compute(alpha, n_cells=5)
        assert (a_low > a_high).all()

    def test_positive_values(self):
        model = SauterMeanInterfacialArea()
        alpha = torch.rand(20, dtype=torch.float64).clamp(0.01, 0.99)
        a = model.compute(alpha, n_cells=20)
        assert (a >= 0).all()

    def test_repr_not_available(self):
        """RTS models don't have __repr__, just check it's a valid object."""
        model = SauterMeanInterfacialArea()
        assert model.d32_0 > 0


class TestBreakupCoalescenceInterfacialArea:
    """Tests for BreakupCoalescenceInterfacialArea."""

    def test_registered(self):
        assert "breakupCoalescence" in InterfacialAreaModel.available_types()

    def test_factory_create(self):
        model = InterfacialAreaModel.create("breakupCoalescence")
        assert isinstance(model, BreakupCoalescenceInterfacialArea)

    def test_default_params(self):
        model = BreakupCoalescenceInterfacialArea()
        assert model.d_eq_0 == pytest.approx(1e-3)
        assert model.C_dev == pytest.approx(0.5)
        assert model.alpha_eq == pytest.approx(0.1)

    def test_zero_alpha_gives_zero_area(self):
        model = BreakupCoalescenceInterfacialArea()
        alpha = torch.zeros(5, dtype=torch.float64)
        a = model.compute(alpha, n_cells=5)
        assert torch.allclose(a, torch.zeros(5, dtype=torch.float64))

    def test_equilibrium_minimum(self):
        """Area should increase more slowly near alpha_eq due to deviation."""
        model = BreakupCoalescenceInterfacialArea(d_eq_0=1e-3, C_dev=1.0, alpha_eq=0.3, alpha_min=1e-10)
        # Use exact alpha values including alpha_eq
        alpha = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=torch.float64)
        a = model.compute(alpha, n_cells=8)

        # At alpha_eq=0.3, deviation = 0, so a = 6 * 0.3 / 1e-3 = 1800
        a_at_eq = float(a[2].item())
        assert abs(a_at_eq - 1800.0) < 1.0

        # Deviation should increase area away from alpha_eq
        # At alpha=0.6: a = 6 * 0.6 / 1e-3 * (1 + (0.6-0.3)^2) = 3600 * 1.09 = 3924
        a_far = float(a[5].item())
        expected_far = 6.0 * 0.6 / 1e-3 * (1.0 + 1.0 * (0.6 - 0.3)**2)
        assert abs(a_far - expected_far) < 1.0

    def test_positive_values(self):
        model = BreakupCoalescenceInterfacialArea()
        alpha = torch.rand(20, dtype=torch.float64).clamp(0.01, 0.99)
        a = model.compute(alpha, n_cells=20)
        assert (a >= 0).all()

    def test_with_epsilon_kwarg(self):
        """Should accept epsilon, sigma, rho_c kwargs."""
        model = BreakupCoalescenceInterfacialArea(alpha_min=1e-10)
        alpha = torch.full((5,), 0.2, dtype=torch.float64)
        epsilon = torch.full((5,), 0.01, dtype=torch.float64)
        a = model.compute(alpha, n_cells=5, epsilon=epsilon, sigma=0.072, rho_c=1000.0)
        assert a.shape == (5,)
        assert (a >= 0).all()


class TestBlendedInterfacialArea:
    """Tests for BlendedInterfacialArea."""

    def test_not_registered(self):
        """Blended model is not RTS-registered (direct use only)."""
        assert "blended" not in InterfacialAreaModel.available_types()

    def test_creation(self):
        m1 = SauterMeanInterfacialArea(d32_0=3e-3, alpha_min=1e-10)
        m2 = BreakupCoalescenceInterfacialArea(d_eq_0=1e-3, alpha_min=1e-10)
        model = BlendedInterfacialArea(m1, m2, alpha_blend=0.3)
        assert model.model_1 is m1
        assert model.model_2 is m2
        assert model.alpha_blend == pytest.approx(0.3)

    def test_dilute_dominates_at_low_alpha(self):
        """At low alpha, result should be close to model_1."""
        m1 = SauterMeanInterfacialArea(d32_0=1e-3, alpha_min=1e-10)
        m2 = BreakupCoalescenceInterfacialArea(d_eq_0=1e-3, alpha_min=1e-10)
        model = BlendedInterfacialArea(m1, m2, alpha_blend=0.5, blend_width=0.01)

        alpha = torch.tensor([0.05], dtype=torch.float64)
        a_blended = model.compute(alpha, n_cells=1)
        a_m1 = m1.compute(alpha, n_cells=1)
        assert torch.allclose(a_blended, a_m1, rtol=0.05)

    def test_dense_dominates_at_high_alpha(self):
        """At high alpha, result should be close to model_2."""
        m1 = SauterMeanInterfacialArea(d32_0=1e-3, alpha_min=1e-10)
        m2 = BreakupCoalescenceInterfacialArea(d_eq_0=1e-3, alpha_min=1e-10)
        model = BlendedInterfacialArea(m1, m2, alpha_blend=0.3, blend_width=0.01)

        alpha = torch.tensor([0.8], dtype=torch.float64)
        a_blended = model.compute(alpha, n_cells=1)
        a_m2 = m2.compute(alpha, n_cells=1)
        assert torch.allclose(a_blended, a_m2, rtol=0.05)

    def test_smooth_transition(self):
        """Result should transition smoothly between models."""
        m1 = ConstantInterfacialAreaForTest(a_i0=100.0)
        m2 = ConstantInterfacialAreaForTest(a_i0=200.0)
        model = BlendedInterfacialArea(m1, m2, alpha_blend=0.5, blend_width=0.1)

        alpha = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
        a = model.compute(alpha, n_cells=50)

        # Should start near 100 and end near 200
        assert float(a[0].item()) < 110.0
        assert float(a[-1].item()) > 190.0
        # Monotonically increasing
        diffs = a[1:] - a[:-1]
        assert (diffs >= -1e-6).all()


class ConstantInterfacialAreaForTest:
    """Simple helper for testing blended model (not RTS-registered)."""

    def __init__(self, a_i0=100.0):
        self._a_i0 = a_i0

    def compute(self, alpha, n_cells, **kwargs):
        return torch.full((n_cells,), self._a_i0, dtype=alpha.dtype)
