"""
Unit tests for Lagrangian evaporation models.
"""

from __future__ import annotations

import math
import pytest

from pyfoam.lagrangian.evaporation import (
    EvaporationModel,
    NoEvaporation,
    RanzMarshallEvaporation,
)


# ======================================================================
# EvaporationModel 抽象基类
# ======================================================================

class TestEvaporationModelABC:
    """Tests for the EvaporationModel abstract base."""

    def test_cannot_instantiate(self):
        """EvaporationModel is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            EvaporationModel()


# ======================================================================
# NoEvaporation
# ======================================================================

class TestNoEvaporation:
    """Tests for NoEvaporation."""

    def test_returns_zero(self):
        model = NoEvaporation()
        dm = model.evaporate(dt=0.01, diameter=1e-3, temperature=350.0,
                             fluid_temperature=500.0)
        assert dm == 0.0

    def test_returns_zero_any_conditions(self):
        """Zero evaporation regardless of conditions."""
        model = NoEvaporation()
        for dT in [10.0, 100.0, 1000.0]:
            dm = model.evaporate(
                dt=0.01, diameter=1e-3,
                temperature=300.0 + dT,
                fluid_temperature=500.0,
            )
            assert dm == 0.0

    def test_returns_zero_any_dt(self):
        model = NoEvaporation()
        for dt in [1e-10, 1e-3, 1.0, 100.0]:
            dm = model.evaporate(dt=dt, diameter=1e-3, temperature=350.0,
                                 fluid_temperature=500.0)
            assert dm == 0.0


# ======================================================================
# RanzMarshallEvaporation
# ======================================================================

class TestRanzMarshallEvaporation:
    """Tests for RanzMarshallEvaporation."""

    def test_zero_diameter_returns_zero(self):
        """Zero diameter → no mass available for evaporation."""
        model = RanzMarshallEvaporation()
        dm = model.evaporate(dt=0.01, diameter=0.0, temperature=350.0,
                             fluid_temperature=500.0)
        assert dm == 0.0

    def test_negligible_diameter_returns_zero(self):
        """Very small diameter → returns zero."""
        model = RanzMarshallEvaporation()
        dm = model.evaporate(dt=0.01, diameter=1e-20, temperature=350.0,
                             fluid_temperature=500.0)
        assert dm == 0.0

    def test_hotter_droplet_no_evaporation(self):
        """When droplet is hotter than fluid, no evaporation."""
        model = RanzMarshallEvaporation()
        dm = model.evaporate(dt=0.01, diameter=1e-3, temperature=600.0,
                             fluid_temperature=400.0)
        assert dm == 0.0

    def test_equal_temperature_no_evaporation(self):
        """When temperatures are equal, no driving force."""
        model = RanzMarshallEvaporation()
        dm = model.evaporate(dt=0.01, diameter=1e-3, temperature=400.0,
                             fluid_temperature=400.0)
        assert dm == 0.0

    def test_evaporation_gives_positive_mass_loss(self):
        """With fluid hotter than droplet, evaporation occurs."""
        model = RanzMarshallEvaporation()
        dm = model.evaporate(
            dt=1e-3,
            diameter=1e-3,
            temperature=350.0,
            fluid_temperature=500.0,
        )
        assert dm > 0.0

    def test_finite_result(self):
        """Result should be finite."""
        model = RanzMarshallEvaporation()
        dm = model.evaporate(
            dt=1e-3,
            diameter=1e-3,
            temperature=350.0,
            fluid_temperature=500.0,
        )
        assert math.isfinite(dm)

    def test_larger_dt_gives_more_evaporation(self):
        """Larger time step should yield more mass loss."""
        model = RanzMarshallEvaporation()
        dm_small = model.evaporate(dt=1e-4, diameter=1e-3, temperature=350.0,
                                   fluid_temperature=500.0)
        dm_large = model.evaporate(dt=1e-2, diameter=1e-3, temperature=350.0,
                                   fluid_temperature=500.0)
        assert dm_large > dm_small

    def test_larger_droplet_gives_more_evaporation(self):
        """Larger droplet has more surface area → more evaporation rate."""
        m_small = RanzMarshallEvaporation()
        m_large = RanzMarshallEvaporation()
        dm_small = m_small.evaporate(dt=1e-3, diameter=1e-4,
                                     temperature=350.0,
                                     fluid_temperature=500.0)
        dm_large = m_large.evaporate(dt=1e-3, diameter=1e-3,
                                     temperature=350.0,
                                     fluid_temperature=500.0)
        assert dm_large > dm_small

    def test_larger_temperature_difference_gives_more_evaporation(self):
        """Larger dT → larger driving force → more evaporation."""
        m_low = RanzMarshallEvaporation()
        m_high = RanzMarshallEvaporation()
        dm_low = m_low.evaporate(dt=1e-3, diameter=1e-3, temperature=390.0,
                                 fluid_temperature=400.0)
        dm_high = m_high.evaporate(dt=1e-3, diameter=1e-3, temperature=300.0,
                                   fluid_temperature=500.0)
        assert dm_high > dm_low

    def test_with_reynolds_number_gives_more_evaporation(self):
        """Non-zero Re adds convective enhancement → more evaporation."""
        m_quiescent = RanzMarshallEvaporation(reynolds_number=0.0)
        m_flow = RanzMarshallEvaporation(reynolds_number=100.0)
        dm_qui = m_quiescent.evaporate(dt=1e-3, diameter=1e-3,
                                        temperature=350.0,
                                        fluid_temperature=500.0)
        dm_flow = m_flow.evaporate(dt=1e-3, diameter=1e-3,
                                   temperature=350.0,
                                   fluid_temperature=500.0)
        assert dm_flow > dm_qui

    def test_negative_reynolds_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            RanzMarshallEvaporation(reynolds_number=-1.0)

    def test_repr(self):
        model = RanzMarshallEvaporation(reynolds_number=50.0)
        r = repr(model)
        assert "RanzMarshallEvaporation" in r
        assert "50" in r
