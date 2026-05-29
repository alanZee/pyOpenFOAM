"""
Unit tests for SprayFoam2 — enhanced Lagrangian spray with KH-RT breakup.

Tests cover:
- WaveBreakupModel initialisation
- KH wavelength and growth rate computation
- RT growth rate computation
- Breakup diameter computation (no-breakup case)
- EnhancedLagrangianCoupling evaporation rate
- Solver import
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


# ---------------------------------------------------------------------------
# WaveBreakupModel tests (no mesh)
# ---------------------------------------------------------------------------


class TestWaveBreakupModel:
    """Tests for KH-RT breakup model."""

    def test_creation(self):
        from pyfoam.applications.spray_foam_2 import WaveBreakupModel
        model = WaveBreakupModel(B0=0.61, B1=1.73, C_rt=1.0)
        assert model.B0 == 0.61
        assert model.B1 == 1.73

    def test_kh_wavelength_positive(self):
        from pyfoam.applications.spray_foam_2 import WaveBreakupModel
        model = WaveBreakupModel(mu=1.8e-5, sigma=0.07, rho_c=1.225)
        lam = model.kh_wavelength(d=1e-3, We=10.0)
        assert lam > 0.0
        assert math.isfinite(lam)

    def test_kh_growth_rate_positive(self):
        from pyfoam.applications.spray_foam_2 import WaveBreakupModel
        model = WaveBreakupModel(sigma=0.07, rho_c=1.225)
        omega = model.kh_growth_rate(d=1e-3, We=10.0)
        assert omega > 0.0
        assert math.isfinite(omega)

    def test_rt_growth_rate_positive(self):
        from pyfoam.applications.spray_foam_2 import WaveBreakupModel
        model = WaveBreakupModel(sigma=0.07, rho_c=1.225)
        omega = model.rt_growth_rate(d=1e-3, a_rel=9.81, rho_d=1000.0)
        assert omega >= 0.0
        assert math.isfinite(omega)

    def test_rt_growth_rate_zero_acceleration(self):
        from pyfoam.applications.spray_foam_2 import WaveBreakupModel
        model = WaveBreakupModel(sigma=0.07, rho_c=1.225)
        omega = model.rt_growth_rate(d=1e-3, a_rel=0.0, rho_d=1000.0)
        assert omega == 0.0

    def test_rt_growth_rate_zero_sigma(self):
        from pyfoam.applications.spray_foam_2 import WaveBreakupModel
        model = WaveBreakupModel(sigma=0.0, rho_c=1.225)
        omega = model.rt_growth_rate(d=1e-3, a_rel=9.81, rho_d=1000.0)
        assert omega == 0.0


class TestBreakupComputation:
    """Tests for breakup diameter computation."""

    def test_no_breakup_for_low_weber(self):
        """Low We droplets should not break up."""
        from pyfoam.applications.spray_foam_2 import WaveBreakupModel
        from pyfoam.lagrangian.particle import Particle

        model = WaveBreakupModel(mu=1.8e-5, sigma=0.07, rho_c=1.225)
        # Very small velocity -> low We
        p = Particle(
            position=[0.5, 0.5, 0.5],
            velocity=[0.001, 0.0, 0.0],
            diameter=1e-3,
            density=1000.0,
        )
        d_new = model.compute_breakup(p, a_rel=9.81)
        # Should be unchanged (no breakup for We < 1)
        assert d_new == p.diameter

    def test_breakup_returns_finite(self):
        from pyfoam.applications.spray_foam_2 import WaveBreakupModel
        from pyfoam.lagrangian.particle import Particle

        model = WaveBreakupModel(mu=1.8e-5, sigma=0.07, rho_c=1.225)
        p = Particle(
            position=[0.5, 0.5, 0.5],
            velocity=[100.0, 0.0, 0.0],
            diameter=1e-3,
            density=1000.0,
        )
        d_new = model.compute_breakup(p, a_rel=9.81)
        assert math.isfinite(d_new)
        assert d_new > 0.0


class TestEnhancedCoupling:
    """Tests for enhanced Lagrangian coupling."""

    def test_saturation_vapour_fraction(self):
        from pyfoam.applications.spray_foam_2 import EnhancedLagrangianCoupling
        from pyfoam.lagrangian.cloud import KinematicCloud
        from unittest.mock import MagicMock

        mesh = MagicMock()
        mesh.n_cells = 10
        cloud = KinematicCloud(
            fluid_velocity=[0.0, 0.0, 0.0],
            fluid_density=1.225,
            fluid_viscosity=1.8e-5,
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 1.0],
        )

        coupling = EnhancedLagrangianCoupling(
            mesh=mesh, cloud=cloud, Cp_fuel=2000.0, L_vap=2.5e6,
        )

        Y_s = coupling._saturation_vapour_fraction(300.0)
        assert 0.0 <= Y_s <= 1.0
        assert math.isfinite(Y_s)

    def test_saturation_increases_with_temperature(self):
        from pyfoam.applications.spray_foam_2 import EnhancedLagrangianCoupling
        from pyfoam.lagrangian.cloud import KinematicCloud
        from unittest.mock import MagicMock

        mesh = MagicMock()
        cloud = KinematicCloud(
            fluid_velocity=[0.0, 0.0, 0.0],
            fluid_density=1.225,
            fluid_viscosity=1.8e-5,
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 1.0],
        )

        coupling = EnhancedLagrangianCoupling(
            mesh=mesh, cloud=cloud, Cp_fuel=2000.0, L_vap=2.5e6,
        )

        Y_300 = coupling._saturation_vapour_fraction(300.0)
        Y_500 = coupling._saturation_vapour_fraction(500.0)
        assert Y_500 > Y_300


class TestSprayFoam2Import:
    """Import tests."""

    def test_imports(self):
        from pyfoam.applications.spray_foam_2 import SprayFoam2
        assert SprayFoam2 is not None

    def test_exports_in_all(self):
        from pyfoam.applications import SprayFoam2
        assert SprayFoam2 is not None
