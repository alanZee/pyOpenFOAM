"""
Unit tests for Lagrangian spray models.
"""

from __future__ import annotations

import math
import pytest

from pyfoam.lagrangian.spray_models import (
    SprayModel,
    BlobAtomization,
    TABBreakup,
)


# ======================================================================
# SprayModel ABC
# ======================================================================

class TestSprayModelABC:
    """Tests for SprayModel abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SprayModel()


# ======================================================================
# BlobAtomization
# ======================================================================

class TestBlobAtomization:
    """Tests for BlobAtomization model."""

    def test_default_parameters(self):
        model = BlobAtomization()
        assert model.blob_diameter == pytest.approx(1e-3)
        assert model.we_crit == pytest.approx(12.0)
        assert model.b0 == pytest.approx(0.61)
        assert model.b1 == pytest.approx(10.0)

    def test_custom_parameters(self):
        model = BlobAtomization(blob_diameter=2e-3, we_crit=6.0)
        assert model.blob_diameter == pytest.approx(2e-3)
        assert model.we_crit == pytest.approx(6.0)

    def test_blob_diameter_must_be_positive(self):
        with pytest.raises(ValueError, match="blob_diameter"):
            BlobAtomization(blob_diameter=0.0)

    def test_we_crit_must_be_positive(self):
        with pytest.raises(ValueError, match="we_crit"):
            BlobAtomization(we_crit=0.0)

    def test_b0_must_be_positive(self):
        with pytest.raises(ValueError, match="b0"):
            BlobAtomization(b0=0.0)

    def test_b1_must_be_positive(self):
        with pytest.raises(ValueError, match="b1"):
            BlobAtomization(b1=0.0)

    def test_zero_diameter_no_atomization(self):
        model = BlobAtomization()
        result = model.atomize(
            dt=1e-5, diameter=0.0, relative_velocity=100.0,
        )
        assert result["atomized"] is False

    def test_zero_velocity_no_atomization(self):
        model = BlobAtomization()
        result = model.atomize(
            dt=1e-5, diameter=1e-3, relative_velocity=0.0,
        )
        assert result["atomized"] is False

    def test_low_weber_no_atomization(self):
        """We < We_crit: no atomization."""
        model = BlobAtomization(we_crit=12.0)
        # We = rho_f * v^2 * r / sigma
        # We = 1.225 * 1^2 * 0.5e-3 / 0.072 = 0.0085 << 12
        result = model.atomize(
            dt=1e-5, diameter=1e-3, relative_velocity=1.0,
            fluid_density=1.225, surface_tension=0.072,
        )
        assert result["atomized"] is False
        assert result["diameter"] == pytest.approx(1e-3)

    def test_high_weber_atomizes(self):
        """We > We_crit: atomization occurs."""
        model = BlobAtomization(we_crit=12.0)
        # We = rho_f * v^2 * r / sigma
        # v=500, d=1e-3: We = 1.225 * 250000 * 0.5e-3 / 0.072 = 2126 >> 12
        result = model.atomize(
            dt=1e-5, diameter=1e-3, relative_velocity=500.0,
            fluid_density=1.225, surface_tension=0.072,
        )
        assert result["atomized"] is True
        assert result["diameter"] < 1e-3
        assert result["diameter"] > 0.0

    def test_atomized_diameter_finite(self):
        model = BlobAtomization()
        result = model.atomize(
            dt=1e-5, diameter=1e-3, relative_velocity=500.0,
            fluid_density=1.225, surface_tension=0.072,
        )
        if result["atomized"]:
            assert math.isfinite(result["diameter"])

    def test_zero_surface_tension_no_atomization(self):
        model = BlobAtomization()
        result = model.atomize(
            dt=1e-5, diameter=1e-3, relative_velocity=500.0,
            surface_tension=0.0,
        )
        assert result["atomized"] is False

    def test_repr(self):
        model = BlobAtomization(blob_diameter=2e-3)
        r = repr(model)
        assert "BlobAtomization" in r
        assert "2e-3" in r or "0.002" in r


# ======================================================================
# TABBreakup
# ======================================================================

class TestTABBreakup:
    """Tests for TABBreakup model."""

    def test_default_parameters(self):
        model = TABBreakup()
        assert model.k_tab == pytest.approx(10.0)
        assert model.c_tab == pytest.approx(0.5)
        assert model.we_crit == pytest.approx(6.0)

    def test_custom_parameters(self):
        model = TABBreakup(k_tab=20.0, c_tab=1.0, we_crit=12.0)
        assert model.k_tab == pytest.approx(20.0)
        assert model.c_tab == pytest.approx(1.0)
        assert model.we_crit == pytest.approx(12.0)

    def test_k_tab_must_be_positive(self):
        with pytest.raises(ValueError, match="k_tab"):
            TABBreakup(k_tab=0.0)

    def test_c_tab_must_be_non_negative(self):
        with pytest.raises(ValueError, match="c_tab"):
            TABBreakup(c_tab=-1.0)

    def test_we_crit_must_be_positive(self):
        with pytest.raises(ValueError, match="we_crit"):
            TABBreakup(we_crit=0.0)

    def test_zero_diameter_no_atomization(self):
        model = TABBreakup()
        result = model.atomize(
            dt=1e-4, diameter=0.0, relative_velocity=100.0,
        )
        assert result["atomized"] is False

    def test_zero_velocity_no_atomization(self):
        model = TABBreakup()
        result = model.atomize(
            dt=1e-4, diameter=1e-3, relative_velocity=0.0,
        )
        assert result["atomized"] is False

    def test_low_weber_no_atomization(self):
        """We < We_crit: no breakup."""
        model = TABBreakup(we_crit=6.0)
        # We = rho_f * v^2 * d / (2 * sigma)
        # v=1, d=1e-3: We = 1.225 * 1 * 1e-3 / (2 * 0.072) = 0.0085 << 6
        result = model.atomize(
            dt=1e-4, diameter=1e-3, relative_velocity=1.0,
            fluid_density=1.225, surface_tension=0.072,
        )
        assert result["atomized"] is False

    def test_high_weber_breakup(self):
        """High We with large dt should produce breakup."""
        model = TABBreakup(k_tab=1.0, we_crit=6.0)
        # We = rho_f * v^2 * d / (2*sigma)
        # v=500, d=1e-3: We = 1.225 * 250000 * 1e-3 / 0.144 = 2126 >> 6
        result = model.atomize(
            dt=0.1, diameter=1e-3, relative_velocity=500.0,
            fluid_density=1.225, surface_tension=0.072,
        )
        assert result["atomized"] is True
        assert result["diameter"] < 1e-3
        assert result["diameter"] > 0.0

    def test_compute_displacement(self):
        model = TABBreakup(k_tab=1.0, we_crit=6.0)
        y = model.compute_displacement(
            dt=0.1, diameter=1e-3, relative_velocity=500.0,
            fluid_density=1.225, surface_tension=0.072,
        )
        assert y > 0.0
        assert math.isfinite(y)

    def test_displacement_zero_at_low_we(self):
        model = TABBreakup(we_crit=6.0)
        y = model.compute_displacement(
            dt=0.1, diameter=1e-3, relative_velocity=1.0,
            fluid_density=1.225, surface_tension=0.072,
        )
        assert y == pytest.approx(0.0)

    def test_larger_dt_more_breakup(self):
        model = TABBreakup(k_tab=1.0, we_crit=6.0)
        common = dict(
            diameter=1e-3, relative_velocity=500.0,
            fluid_density=1.225, surface_tension=0.072,
        )
        r_small = model.atomize(dt=1e-6, **common)
        r_large = model.atomize(dt=0.1, **common)
        if r_small["atomized"] and r_large["atomized"]:
            assert r_large["diameter"] <= r_small["diameter"]

    def test_zero_surface_tension_no_atomization(self):
        model = TABBreakup()
        result = model.atomize(
            dt=0.1, diameter=1e-3, relative_velocity=500.0,
            surface_tension=0.0,
        )
        assert result["atomized"] is False

    def test_repr(self):
        model = TABBreakup(k_tab=20.0)
        r = repr(model)
        assert "TABBreakup" in r
        assert "20" in r
