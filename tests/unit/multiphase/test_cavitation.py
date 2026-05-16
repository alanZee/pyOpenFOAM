"""
Unit tests for cavitation models (Schnerr-Sauer, Merkle, ZGB).

Tests verify:
- Mass transfer direction (evaporation at low pressure, condensation at high)
- Equilibrium at vapor pressure
- Finite output values
"""

from __future__ import annotations

import math

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


class TestSchnerrSauer:
    """Tests for Schnerr-Sauer cavitation model."""

    def test_init(self):
        from pyfoam.multiphase.cavitation import SchnerrSauer

        model = SchnerrSauer(n_b=1e13, p_v=2300.0)
        assert model.n_b == 1e13
        assert model.p_v == 2300.0

    def test_evaporation_at_low_pressure(self):
        """At p < p_v, mass transfer should be positive (evaporation)."""
        from pyfoam.multiphase.cavitation import SchnerrSauer

        model = SchnerrSauer(n_b=1e13, p_v=2300.0)

        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        p = torch.full((10,), 1000.0, dtype=CFD_DTYPE)  # below p_v

        m_dot = model.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        assert (m_dot >= 0).all(), "Evaporation should produce positive m_dot"

    def test_condensation_at_high_pressure(self):
        """At p > p_v, mass transfer should be negative (condensation)."""
        from pyfoam.multiphase.cavitation import SchnerrSauer

        model = SchnerrSauer(n_b=1e13, p_v=2300.0)

        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        p = torch.full((10,), 5000.0, dtype=CFD_DTYPE)  # above p_v

        m_dot = model.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        assert (m_dot <= 0).all(), "Condensation should produce negative m_dot"

    def test_equilibrium_at_vapor_pressure(self):
        """At p = p_v, mass transfer should be approximately zero."""
        from pyfoam.multiphase.cavitation import SchnerrSauer

        model = SchnerrSauer(n_b=1e13, p_v=2300.0)

        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        p = torch.full((10,), 2300.0, dtype=CFD_DTYPE)

        m_dot = model.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        assert torch.allclose(m_dot, torch.zeros_like(m_dot), atol=1e-20)

    def test_finite_output(self):
        """Output is always finite."""
        from pyfoam.multiphase.cavitation import SchnerrSauer

        model = SchnerrSauer(n_b=1e13, p_v=2300.0)

        alpha = torch.rand(20, dtype=CFD_DTYPE).clamp(0.01, 0.99)
        p = torch.randn(20, dtype=CFD_DTYPE) * 1000 + 2300

        m_dot = model.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        assert torch.isfinite(m_dot).all()

    def test_magnitude_increases_with_pressure_difference(self):
        """Larger pressure difference produces larger mass transfer."""
        from pyfoam.multiphase.cavitation import SchnerrSauer

        model = SchnerrSauer(n_b=1e13, p_v=2300.0)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)

        p_low = torch.full((10,), 500.0, dtype=CFD_DTYPE)
        p_high = torch.full((10,), 100.0, dtype=CFD_DTYPE)

        m_dot_low = model.compute_mass_transfer(alpha, p_low, 1000.0, 0.02)
        m_dot_high = model.compute_mass_transfer(alpha, p_high, 1000.0, 0.02)

        assert m_dot_high.abs().mean() > m_dot_low.abs().mean()


class TestMerkle:
    """Tests for Merkle cavitation model."""

    def test_init(self):
        from pyfoam.multiphase.cavitation import Merkle

        model = Merkle(C_evap=1.0, C_cond=1.0, p_v=2300.0)
        assert model.C_evap == 1.0

    def test_evaporation_condensation(self):
        from pyfoam.multiphase.cavitation import Merkle

        model = Merkle(p_v=2300.0)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)

        # Below p_v: evaporation
        p_evap = torch.full((10,), 1000.0, dtype=CFD_DTYPE)
        m_dot = model.compute_mass_transfer(alpha, p_evap, 1000.0, 0.02)
        assert (m_dot >= 0).all()

        # Above p_v: condensation
        p_cond = torch.full((10,), 5000.0, dtype=CFD_DTYPE)
        m_dot = model.compute_mass_transfer(alpha, p_cond, 1000.0, 0.02)
        assert (m_dot <= 0).all()


class TestZGB:
    """Tests for Zwart-Gerber-Belamri cavitation model."""

    def test_init(self):
        from pyfoam.multiphase.cavitation import ZGB

        model = ZGB(C_evap=0.02, C_cond=0.01, p_v=2300.0)
        assert model.C_evap == 0.02

    def test_evaporation_condensation(self):
        from pyfoam.multiphase.cavitation import ZGB

        model = ZGB(p_v=2300.0)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)

        # Below p_v: evaporation (positive m_dot)
        p_evap = torch.full((10,), 1000.0, dtype=CFD_DTYPE)
        m_dot = model.compute_mass_transfer(alpha, p_evap, 1000.0, 0.02)
        assert (m_dot >= 0).all()

        # Above p_v: condensation (negative m_dot)
        p_cond = torch.full((10,), 5000.0, dtype=CFD_DTYPE)
        m_dot = model.compute_mass_transfer(alpha, p_cond, 1000.0, 0.02)
        assert (m_dot <= 0).all()

    def test_finite_output(self):
        from pyfoam.multiphase.cavitation import ZGB

        model = ZGB(p_v=2300.0)
        alpha = torch.rand(20, dtype=CFD_DTYPE).clamp(0.01, 0.99)
        p = torch.randn(20, dtype=CFD_DTYPE) * 1000 + 2300

        m_dot = model.compute_mass_transfer(alpha, p, 1000.0, 0.02)
        assert torch.isfinite(m_dot).all()
