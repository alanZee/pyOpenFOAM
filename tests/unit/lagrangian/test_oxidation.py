"""
Unit tests for Lagrangian oxidation models.
"""

from __future__ import annotations

import math
import pytest

from pyfoam.lagrangian.oxidation import (
    OxidationModel,
    NoOxidation,
    FieldOxidation,
)


# ======================================================================
# OxidationModel 抽象基类
# ======================================================================

class TestOxidationModelABC:
    """Tests for the OxidationModel abstract base."""

    def test_cannot_instantiate(self):
        """OxidationModel is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            OxidationModel()


# ======================================================================
# NoOxidation
# ======================================================================

class TestNoOxidation:
    """Tests for NoOxidation."""

    def test_returns_zero(self):
        model = NoOxidation()
        dm = model.oxidise(dt=0.01, diameter=1e-3, temperature=1500.0)
        assert dm == 0.0

    def test_returns_zero_any_conditions(self):
        """Zero oxidation regardless of conditions."""
        model = NoOxidation()
        for T in [500.0, 1500.0, 3000.0]:
            dm = model.oxidise(
                dt=0.01, diameter=1e-3,
                temperature=T, oxygen_mass_fraction=0.23,
            )
            assert dm == 0.0

    def test_returns_zero_any_dt(self):
        model = NoOxidation()
        for dt in [1e-10, 1e-3, 1.0, 100.0]:
            dm = model.oxidise(dt=dt, diameter=1e-3, temperature=1500.0)
            assert dm == 0.0


# ======================================================================
# FieldOxidation
# ======================================================================

class TestFieldOxidation:
    """Tests for FieldOxidation (Arrhenius oxidation)."""

    def test_zero_diameter_returns_zero(self):
        """Zero diameter → no surface for reaction."""
        model = FieldOxidation()
        dm = model.oxidise(dt=0.01, diameter=0.0, temperature=1500.0)
        assert dm == 0.0

    def test_negligible_diameter_returns_zero(self):
        """Very small diameter → returns zero."""
        model = FieldOxidation()
        dm = model.oxidise(dt=0.01, diameter=1e-20, temperature=1500.0)
        assert dm == 0.0

    def test_zero_oxygen_returns_zero(self):
        """No O₂ → no oxidation."""
        model = FieldOxidation()
        dm = model.oxidise(dt=0.01, diameter=1e-3, temperature=1500.0,
                           oxygen_mass_fraction=0.0)
        assert dm == 0.0

    def test_negligible_oxygen_returns_zero(self):
        """Negligible O₂ → returns zero."""
        model = FieldOxidation()
        dm = model.oxidise(dt=0.01, diameter=1e-3, temperature=1500.0,
                           oxygen_mass_fraction=1e-20)
        assert dm == 0.0

    def test_zero_temperature_returns_zero(self):
        """Zero temperature → Arrhenius rate is zero."""
        model = FieldOxidation()
        dm = model.oxidise(dt=0.01, diameter=1e-3, temperature=0.0)
        assert dm == 0.0

    def test_very_low_temperature_returns_zero(self):
        """Very low temperature → Arrhenius rate is negligibly small."""
        model = FieldOxidation(activation_energy=8e4)
        dm = model.oxidise(dt=0.01, diameter=1e-3, temperature=100.0,
                           oxygen_mass_fraction=0.23)
        assert dm == 0.0

    def test_oxidation_gives_positive_mass_loss(self):
        """At high temperature with O₂, oxidation occurs."""
        model = FieldOxidation()
        dm = model.oxidise(
            dt=1e-3,
            diameter=1e-3,
            temperature=1500.0,
            oxygen_mass_fraction=0.23,
        )
        assert dm > 0.0

    def test_finite_result(self):
        """Result should be finite."""
        model = FieldOxidation()
        dm = model.oxidise(
            dt=1e-3,
            diameter=1e-3,
            temperature=1500.0,
            oxygen_mass_fraction=0.23,
        )
        assert math.isfinite(dm)

    def test_higher_temperature_gives_faster_oxidation(self):
        """Higher temperature → higher Arrhenius rate → more mass loss."""
        m_low = FieldOxidation()
        m_high = FieldOxidation()
        dm_low = m_low.oxidise(dt=1e-3, diameter=1e-3, temperature=1000.0,
                               oxygen_mass_fraction=0.23)
        dm_high = m_high.oxidise(dt=1e-3, diameter=1e-3, temperature=2000.0,
                                 oxygen_mass_fraction=0.23)
        assert dm_high > dm_low

    def test_more_oxygen_gives_faster_oxidation(self):
        """Higher O₂ mass fraction → faster oxidation."""
        m_low = FieldOxidation()
        m_high = FieldOxidation()
        dm_low = m_low.oxidise(dt=1e-3, diameter=1e-3, temperature=1500.0,
                               oxygen_mass_fraction=0.05)
        dm_high = m_high.oxidise(dt=1e-3, diameter=1e-3, temperature=1500.0,
                                 oxygen_mass_fraction=0.23)
        assert dm_high > dm_low

    def test_larger_dt_gives_more_oxidation(self):
        """Larger time step → more mass loss."""
        model = FieldOxidation()
        dm_small = model.oxidise(dt=1e-4, diameter=1e-3, temperature=1500.0,
                                 oxygen_mass_fraction=0.23)
        dm_large = model.oxidise(dt=1e-2, diameter=1e-3, temperature=1500.0,
                                 oxygen_mass_fraction=0.23)
        assert dm_large > dm_small

    def test_larger_particle_gives_more_oxidation(self):
        """Larger particle → more surface area → more mass loss per step."""
        m_small = FieldOxidation()
        m_large = FieldOxidation()
        dm_small = m_small.oxidise(dt=1e-3, diameter=1e-4, temperature=1500.0,
                                   oxygen_mass_fraction=0.23)
        dm_large = m_large.oxidise(dt=1e-3, diameter=1e-3, temperature=1500.0,
                                   oxygen_mass_fraction=0.23)
        assert dm_large > dm_small

    def test_higher_density_gives_faster_oxidation(self):
        """Higher fluid density → more O₂ per unit volume → faster rate."""
        m_low = FieldOxidation()
        m_high = FieldOxidation()
        dm_low = m_low.oxidise(dt=1e-3, diameter=1e-3, temperature=1500.0,
                               oxygen_mass_fraction=0.23, fluid_density=0.5)
        dm_high = m_high.oxidise(dt=1e-3, diameter=1e-3, temperature=1500.0,
                                 oxygen_mass_fraction=0.23, fluid_density=2.0)
        assert dm_high > dm_low

    def test_negative_pre_exponential_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            FieldOxidation(pre_exponential=-1.0)

    def test_negative_activation_energy_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            FieldOxidation(activation_energy=-1.0)

    def test_higher_activation_energy_gives_slower_oxidation(self):
        """Higher Ea → lower Arrhenius rate → less mass loss."""
        m_low_ea = FieldOxidation(activation_energy=5e4)
        m_high_ea = FieldOxidation(activation_energy=2e5)
        dm_low = m_low_ea.oxidise(dt=1e-3, diameter=1e-3, temperature=1200.0,
                                   oxygen_mass_fraction=0.23)
        dm_high = m_high_ea.oxidise(dt=1e-3, diameter=1e-3, temperature=1200.0,
                                    oxygen_mass_fraction=0.23)
        assert dm_low > dm_high

    def test_higher_pre_exponential_gives_faster_oxidation(self):
        """Higher A → higher rate → more mass loss."""
        m_low_a = FieldOxidation(pre_exponential=0.1)
        m_high_a = FieldOxidation(pre_exponential=10.0)
        dm_low = m_low_a.oxidise(dt=1e-3, diameter=1e-3, temperature=1500.0,
                                  oxygen_mass_fraction=0.23)
        dm_high = m_high_a.oxidise(dt=1e-3, diameter=1e-3, temperature=1500.0,
                                   oxygen_mass_fraction=0.23)
        assert dm_high > dm_low

    def test_repr(self):
        model = FieldOxidation(pre_exponential=2.5, activation_energy=1e5)
        r = repr(model)
        assert "FieldOxidation" in r
        assert "2.5" in r
        assert "100000" in r or "1e+05" in r or "1e5" in r
