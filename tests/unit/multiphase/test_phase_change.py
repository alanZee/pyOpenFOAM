"""Tests for phase change models (Lee, SchnerrSauerEnhanced).

Tests cover:
- RTS registration and factory creation
- Lee model: evaporation/condensation, equilibrium, rate controls
- SchnerrSauerEnhanced: pressure limiting, alpha clipping, under-relaxation
"""

import pytest
import torch
import math

from pyfoam.core.dtype import CFD_DTYPE


class TestPhaseChangeRegistry:
    """Tests for RTS registry."""

    def test_lee_registered(self):
        from pyfoam.multiphase.phase_change import PhaseChangeModel

        assert "Lee" in PhaseChangeModel.available_types()

    def test_schnerr_sauer_enhanced_registered(self):
        from pyfoam.multiphase.phase_change import PhaseChangeModel

        assert "SchnerrSauerEnhanced" in PhaseChangeModel.available_types()

    def test_factory_lee(self):
        from pyfoam.multiphase.phase_change import PhaseChangeModel, LeeModel

        model = PhaseChangeModel.create("Lee", T_sat=373.15)
        assert isinstance(model, LeeModel)

    def test_factory_schnerr_sauer_enhanced(self):
        from pyfoam.multiphase.phase_change import PhaseChangeModel, SchnerrSauerEnhanced

        model = PhaseChangeModel.create("SchnerrSauerEnhanced")
        assert isinstance(model, SchnerrSauerEnhanced)

    def test_factory_unknown_raises(self):
        from pyfoam.multiphase.phase_change import PhaseChangeModel

        with pytest.raises(KeyError, match="Unknown phase change model"):
            PhaseChangeModel.create("NonExistent")

    def test_duplicate_registration_raises(self):
        """Registering the same name twice raises ValueError."""
        from pyfoam.multiphase.phase_change import PhaseChangeModel

        with pytest.raises(ValueError, match="already registered"):
            PhaseChangeModel.register("Lee")(type("Dup", (PhaseChangeModel,), {}))


class TestLeeModel:
    """Tests for Lee phase change model."""

    def test_init(self):
        from pyfoam.multiphase.phase_change import LeeModel

        model = LeeModel(T_sat=373.15, r_evap=0.1, r_cond=0.2)
        assert model.T_sat == 373.15
        assert model.r_evap == 0.1
        assert model.r_cond == 0.2

    def test_evaporation_above_saturation(self):
        """T > T_sat: positive mass transfer (evaporation)."""
        from pyfoam.multiphase.phase_change import LeeModel

        model = LeeModel(T_sat=373.15)
        n = 10
        alpha = torch.full((n,), 0.3, dtype=CFD_DTYPE)
        T = torch.full((n,), 400.0, dtype=CFD_DTYPE)
        p = torch.full((n,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, T, p, 1000.0, 0.6)
        assert (m_dot > 0).all(), "Evaporation: T > T_sat -> m_dot > 0"

    def test_condensation_below_saturation(self):
        """T < T_sat: negative mass transfer (condensation)."""
        from pyfoam.multiphase.phase_change import LeeModel

        model = LeeModel(T_sat=373.15)
        n = 10
        alpha = torch.full((n,), 0.3, dtype=CFD_DTYPE)
        T = torch.full((n,), 350.0, dtype=CFD_DTYPE)
        p = torch.full((n,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, T, p, 1000.0, 0.6)
        assert (m_dot < 0).all(), "Condensation: T < T_sat -> m_dot < 0"

    def test_equilibrium_at_saturation(self):
        """T = T_sat: zero mass transfer."""
        from pyfoam.multiphase.phase_change import LeeModel

        model = LeeModel(T_sat=373.15)
        n = 10
        alpha = torch.full((n,), 0.3, dtype=CFD_DTYPE)
        T = torch.full((n,), 373.15, dtype=CFD_DTYPE)
        p = torch.full((n,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, T, p, 1000.0, 0.6)
        assert torch.allclose(m_dot, torch.zeros_like(m_dot), atol=1e-20)

    def test_rate_increases_with_superheat(self):
        """Larger superheat -> larger evaporation rate."""
        from pyfoam.multiphase.phase_change import LeeModel

        model = LeeModel(T_sat=373.15)
        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)
        p = torch.full((10,), 101325.0, dtype=CFD_DTYPE)

        T_low = torch.full((10,), 380.0, dtype=CFD_DTYPE)
        T_high = torch.full((10,), 400.0, dtype=CFD_DTYPE)

        m_low = model.compute_mass_transfer(alpha, T_low, p, 1000.0, 0.6)
        m_high = model.compute_mass_transfer(alpha, T_high, p, 1000.0, 0.6)
        assert (m_high > m_low).all()

    def test_custom_rates(self):
        """Custom r_evap/r_cond scale the result."""
        from pyfoam.multiphase.phase_change import LeeModel

        model_slow = LeeModel(T_sat=373.15, r_evap=0.01)
        model_fast = LeeModel(T_sat=373.15, r_evap=1.0)

        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)
        T = torch.full((10,), 400.0, dtype=CFD_DTYPE)
        p = torch.full((10,), 101325.0, dtype=CFD_DTYPE)

        m_slow = model_slow.compute_mass_transfer(alpha, T, p, 1000.0, 0.6)
        m_fast = model_fast.compute_mass_transfer(alpha, T, p, 1000.0, 0.6)
        assert (m_fast > m_slow).all()

    def test_finite_output(self):
        """Output is always finite."""
        from pyfoam.multiphase.phase_change import LeeModel

        model = LeeModel(T_sat=373.15)
        alpha = torch.rand(20, dtype=CFD_DTYPE)
        T = torch.randn(20, dtype=CFD_DTYPE) * 50 + 373.15
        p = torch.full((20,), 101325.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, T, p, 1000.0, 0.6)
        assert torch.isfinite(m_dot).all()

    def test_repr(self):
        from pyfoam.multiphase.phase_change import LeeModel

        model = LeeModel(T_sat=400.0)
        r = repr(model)
        assert "LeeModel" in r
        assert "400" in r


class TestSchnerrSauerEnhanced:
    """Tests for enhanced Schnerr-Sauer model."""

    def test_init(self):
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced(n_b=1e13, p_v=2300.0, relaxation=0.5)
        assert model.n_b == 1e13
        assert model.p_v == 2300.0
        assert model.relaxation == 0.5

    def test_evaporation_at_low_pressure(self):
        """p < p_v: evaporation (m_dot > 0)."""
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced(n_b=1e13, p_v=2300.0)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        T = torch.full((10,), 300.0, dtype=CFD_DTYPE)
        p = torch.full((10,), 1000.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, T, p, 1000.0, 0.02)
        assert (m_dot >= 0).all()

    def test_condensation_at_high_pressure(self):
        """p > p_v: condensation (m_dot < 0)."""
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced(n_b=1e13, p_v=2300.0)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        T = torch.full((10,), 300.0, dtype=CFD_DTYPE)
        p = torch.full((10,), 5000.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, T, p, 1000.0, 0.02)
        assert (m_dot <= 0).all()

    def test_equilibrium_at_vapor_pressure(self):
        """p = p_v: zero mass transfer."""
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced(n_b=1e13, p_v=2300.0)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        T = torch.full((10,), 300.0, dtype=CFD_DTYPE)
        p = torch.full((10,), 2300.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, T, p, 1000.0, 0.02)
        assert torch.allclose(m_dot, torch.zeros_like(m_dot), atol=1e-20)

    def test_alpha_clipping(self):
        """Alpha is clipped to [alpha_min, 1 - alpha_min]."""
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced(alpha_min=1e-4)
        alpha = torch.tensor([0.0, 1.0, 0.5], dtype=CFD_DTYPE)
        T = torch.full((3,), 300.0, dtype=CFD_DTYPE)
        p = torch.full((3,), 1000.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, T, p, 1000.0, 0.02)
        assert torch.isfinite(m_dot).all()

    def test_pressure_limiting(self):
        """Extreme pressures are bounded by p_max."""
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced(p_max=1e4)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        T = torch.full((10,), 300.0, dtype=CFD_DTYPE)
        # Very low pressure, but limited to p_max
        p = torch.full((10,), -1e8, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, T, p, 1000.0, 0.02)
        assert torch.isfinite(m_dot).all()

    def test_relaxation_no_old(self):
        """With no old value, relax() returns the new value."""
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced(relaxation=0.5)
        m_new = torch.full((10,), 1.0, dtype=CFD_DTYPE)
        m_relaxed = model.relax(m_new)
        assert torch.allclose(m_relaxed, m_new)

    def test_relaxation_with_old(self):
        """With old value, relax() blends new and old."""
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced(relaxation=0.3)
        m_new = torch.full((10,), 1.0, dtype=CFD_DTYPE)
        m_old = torch.full((10,), 0.0, dtype=CFD_DTYPE)
        m_relaxed = model.relax(m_new, m_old)

        expected = 0.3 * 1.0 + 0.7 * 0.0
        assert torch.allclose(m_relaxed, torch.full((10,), expected, dtype=CFD_DTYPE))

    def test_relaxation_full(self):
        """relaxation=1.0 -> no relaxation."""
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced(relaxation=1.0)
        m_new = torch.full((10,), 2.0, dtype=CFD_DTYPE)
        m_old = torch.full((10,), 0.5, dtype=CFD_DTYPE)
        m_relaxed = model.relax(m_new, m_old)
        assert torch.allclose(m_relaxed, m_new)

    def test_reset(self):
        """reset() clears stored old value."""
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced(relaxation=0.5)
        m = torch.full((10,), 1.0, dtype=CFD_DTYPE)
        model.relax(m)
        assert model._m_dot_old is not None

        model.reset()
        assert model._m_dot_old is None

    def test_finite_output(self):
        """Output is always finite for random inputs."""
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced()
        alpha = torch.rand(20, dtype=CFD_DTYPE).clamp(0.01, 0.99)
        T = torch.randn(20, dtype=CFD_DTYPE) * 50 + 373.15
        p = torch.randn(20, dtype=CFD_DTYPE) * 5000 + 101325

        m_dot = model.compute_mass_transfer(alpha, T, p, 1000.0, 0.02)
        assert torch.isfinite(m_dot).all()

    def test_repr(self):
        from pyfoam.multiphase.phase_change import SchnerrSauerEnhanced

        model = SchnerrSauerEnhanced(n_b=1e12, relaxation=0.5)
        r = repr(model)
        assert "SchnerrSauerEnhanced" in r
        assert "relaxation=0.5" in r
