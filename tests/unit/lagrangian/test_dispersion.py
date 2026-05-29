"""
Unit tests for Lagrangian dispersion models.
"""

from __future__ import annotations

import math
import pytest

from pyfoam.lagrangian.dispersion import (
    DispersionModel,
    NoDispersion,
    GradientDispersion,
    StochasticDispersion,
)


# ======================================================================
# DispersionModel 抽象基类
# ======================================================================

class TestDispersionModelABC:
    """Tests for the DispersionModel abstract base."""

    def test_cannot_instantiate(self):
        """DispersionModel is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            DispersionModel()


# ======================================================================
# NoDispersion
# ======================================================================

class TestNoDispersion:
    """Tests for NoDispersion."""

    def test_returns_zero(self):
        model = NoDispersion()
        dv = model.disperse(dt=0.01)
        assert dv == [0.0, 0.0, 0.0]

    def test_returns_zero_with_turbulence(self):
        """Even with non-zero turbulence, no dispersion occurs."""
        model = NoDispersion()
        dv = model.disperse(
            dt=0.01,
            turbulent_kinetic_energy=1.0,
            turbulent_dissipation=0.1,
        )
        assert dv == [0.0, 0.0, 0.0]

    def test_returns_zero_any_dt(self):
        model = NoDispersion()
        for dt in [1e-10, 1e-3, 1.0, 100.0]:
            dv = model.disperse(dt=dt)
            assert dv == [0.0, 0.0, 0.0]


# ======================================================================
# GradientDispersion
# ======================================================================

class TestGradientDispersion:
    """Tests for GradientDispersion."""

    def test_zero_k_returns_zero(self):
        """No turbulence → no dispersion."""
        model = GradientDispersion(seed=42)
        dv = model.disperse(dt=0.01, turbulent_kinetic_energy=0.0,
                            turbulent_dissipation=0.1)
        assert dv == [0.0, 0.0, 0.0]

    def test_zero_epsilon_returns_zero(self):
        """Zero dissipation → no dispersion."""
        model = GradientDispersion(seed=42)
        dv = model.disperse(dt=0.01, turbulent_kinetic_energy=1.0,
                            turbulent_dissipation=0.0)
        assert dv == [0.0, 0.0, 0.0]

    def test_nonzero_turbulence_gives_nonzero_dispersion(self):
        """With turbulence, dispersion should be non-zero."""
        model = GradientDispersion(seed=42)
        dv = model.disperse(
            dt=0.01,
            turbulent_kinetic_energy=1.0,
            turbulent_dissipation=0.1,
        )
        # 至少有一个分量应非零
        assert any(abs(v) > 1e-15 for v in dv)

    def test_finite_values(self):
        """All components should be finite."""
        model = GradientDispersion(seed=42)
        dv = model.disperse(
            dt=0.001,
            turbulent_kinetic_energy=0.5,
            turbulent_dissipation=0.05,
        )
        assert all(math.isfinite(v) for v in dv)

    def test_reproducible_with_seed(self):
        """Same seed should give identical results."""
        m1 = GradientDispersion(seed=123)
        m2 = GradientDispersion(seed=123)
        dv1 = m1.disperse(dt=0.01, turbulent_kinetic_energy=0.5,
                          turbulent_dissipation=0.05)
        dv2 = m2.disperse(dt=0.01, turbulent_kinetic_energy=0.5,
                          turbulent_dissipation=0.05)
        assert dv1 == dv2

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) give different results."""
        m1 = GradientDispersion(seed=1)
        m2 = GradientDispersion(seed=2)
        dv1 = m1.disperse(dt=0.01, turbulent_kinetic_energy=0.5,
                          turbulent_dissipation=0.05)
        dv2 = m2.disperse(dt=0.01, turbulent_kinetic_energy=0.5,
                          turbulent_dissipation=0.05)
        assert dv1 != dv2

    def test_intensity_multiplier(self):
        """Higher intensity should increase perturbation magnitude."""
        m_low = GradientDispersion(intensity=0.1, seed=42)
        m_high = GradientDispersion(intensity=10.0, seed=42)
        k, eps = 0.5, 0.05
        dt = 0.01
        dv_low = m_low.disperse(dt=dt, turbulent_kinetic_energy=k,
                                turbulent_dissipation=eps)
        dv_high = m_high.disperse(dt=dt, turbulent_kinetic_energy=k,
                                  turbulent_dissipation=eps)
        mag_low = math.sqrt(sum(v ** 2 for v in dv_low))
        mag_high = math.sqrt(sum(v ** 2 for v in dv_high))
        assert mag_high > mag_low

    def test_larger_dt_gives_larger_dispersion(self):
        """Larger time step should scale perturbation up."""
        model = GradientDispersion(seed=42)
        k, eps = 0.5, 0.05
        dv_small = model.disperse(dt=0.001, turbulent_kinetic_energy=k,
                                  turbulent_dissipation=eps)
        # Need fresh model with same seed for fair comparison
        model2 = GradientDispersion(seed=42)
        dv_large = model2.disperse(dt=0.1, turbulent_kinetic_energy=k,
                                   turbulent_dissipation=eps)
        mag_small = math.sqrt(sum(v ** 2 for v in dv_small))
        mag_large = math.sqrt(sum(v ** 2 for v in dv_large))
        # sqrt(0.1/0.001) = sqrt(100) = 10× scaling
        assert mag_large > mag_small

    def test_negative_intensity_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            GradientDispersion(intensity=-1.0)

    def test_zero_c_tau_raises(self):
        with pytest.raises(ValueError, match="positive"):
            GradientDispersion(c_tau=0.0)

    def test_negative_c_tau_raises(self):
        with pytest.raises(ValueError, match="positive"):
            GradientDispersion(c_tau=-0.1)

    def test_repr(self):
        model = GradientDispersion(intensity=0.5, c_tau=0.3)
        r = repr(model)
        assert "GradientDispersion" in r
        assert "0.5" in r
        assert "0.3" in r


# ======================================================================
# StochasticDispersion
# ======================================================================

class TestStochasticDispersion:
    """Tests for StochasticDispersion."""

    def test_zero_k_returns_zero(self):
        """No turbulence → no dispersion."""
        model = StochasticDispersion(seed=42)
        dv = model.disperse(dt=0.01, turbulent_kinetic_energy=0.0,
                            turbulent_dissipation=0.1)
        assert dv == [0.0, 0.0, 0.0]

    def test_zero_epsilon_returns_zero(self):
        """Zero dissipation → no dispersion."""
        model = StochasticDispersion(seed=42)
        dv = model.disperse(dt=0.01, turbulent_kinetic_energy=1.0,
                            turbulent_dissipation=0.0)
        assert dv == [0.0, 0.0, 0.0]

    def test_nonzero_turbulence_gives_nonzero_dispersion(self):
        """With turbulence, dispersion should be non-zero after a step."""
        model = StochasticDispersion(seed=42)
        dv = model.disperse(
            dt=0.01,
            turbulent_kinetic_energy=1.0,
            turbulent_dissipation=0.1,
        )
        assert any(abs(v) > 1e-15 for v in dv)

    def test_finite_values(self):
        """All components should be finite."""
        model = StochasticDispersion(seed=42)
        for _ in range(10):
            dv = model.disperse(
                dt=0.001,
                turbulent_kinetic_energy=0.5,
                turbulent_dissipation=0.05,
            )
            assert all(math.isfinite(v) for v in dv)

    def test_reproducible_with_seed(self):
        """Same seed should give identical results."""
        m1 = StochasticDispersion(seed=123)
        m2 = StochasticDispersion(seed=123)
        dv1 = m1.disperse(dt=0.01, turbulent_kinetic_energy=0.5,
                          turbulent_dissipation=0.05)
        dv2 = m2.disperse(dt=0.01, turbulent_kinetic_energy=0.5,
                          turbulent_dissipation=0.05)
        assert dv1 == dv2

    def test_temporal_correlation(self):
        """Successive calls should show temporal correlation (stateful)."""
        model = StochasticDispersion(seed=42)
        # 连续调用应返回不同的结果（因为时间演化）
        dv1 = model.disperse(dt=0.01, turbulent_kinetic_energy=0.5,
                             turbulent_dissipation=0.05)
        dv2 = model.disperse(dt=0.01, turbulent_kinetic_energy=0.5,
                             turbulent_dissipation=0.05)
        # 两次结果不同（O-U 过程的随机性）
        assert dv1 != dv2

    def test_reset_zeroes_state(self):
        """reset() should zero out the velocity fluctuation."""
        model = StochasticDispersion(seed=42)
        model.disperse(dt=0.01, turbulent_kinetic_energy=1.0,
                       turbulent_dissipation=0.1)
        model.reset()
        # reset 后下一个输出应该接近零（因为 u_prime 已归零）
        dv = model.disperse(dt=0.0, turbulent_kinetic_energy=1.0,
                            turbulent_dissipation=0.1)
        assert dv == [0.0, 0.0, 0.0]

    def test_negative_c_tau_raises(self):
        with pytest.raises(ValueError, match="positive"):
            StochasticDispersion(c_tau=-0.1)

    def test_zero_c_tau_raises(self):
        with pytest.raises(ValueError, match="positive"):
            StochasticDispersion(c_tau=0.0)

    def test_repr(self):
        model = StochasticDispersion(c_tau=0.4)
        r = repr(model)
        assert "StochasticDispersion" in r
        assert "0.4" in r
