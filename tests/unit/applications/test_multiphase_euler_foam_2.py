"""
Unit tests for MultiphaseEulerFoam2 — enhanced with population balance.

Tests cover:
- SizeGroup dataclass
- Luo breakage kernel
- Prince-Blanch coalescence kernel
- Solver import
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


# ---------------------------------------------------------------------------
# SizeGroup tests
# ---------------------------------------------------------------------------


class TestSizeGroup:
    """Tests for SizeGroup dataclass."""

    def test_size_group_creation(self):
        from pyfoam.applications.multiphase_euler_foam_2 import SizeGroup
        sg = SizeGroup(index=0, diameter=1e-3)
        assert sg.index == 0
        assert sg.diameter == 1e-3
        assert sg.fraction == 0.0

    def test_volume_auto_computed(self):
        from pyfoam.applications.multiphase_euler_foam_2 import SizeGroup
        sg = SizeGroup(index=0, diameter=1e-3)
        expected = math.pi / 6.0 * (1e-3) ** 3
        assert abs(sg.volume - expected) < 1e-30


class TestLuoBreakageKernel:
    """Tests for Luo breakage rate kernel."""

    def test_zero_for_zero_dissipation(self):
        from pyfoam.applications.multiphase_euler_foam_2 import luo_breakage_rate
        rate = luo_breakage_rate(d_i=1e-3, epsilon=0.0, sigma=0.07,
                                  rho_c=1.225, rho_d=1000.0)
        assert rate == 0.0

    def test_zero_for_zero_sigma(self):
        from pyfoam.applications.multiphase_euler_foam_2 import luo_breakage_rate
        rate = luo_breakage_rate(d_i=1e-3, epsilon=0.01, sigma=0.0,
                                  rho_c=1.225, rho_d=1000.0)
        assert rate == 0.0

    def test_positive_for_physical_params(self):
        from pyfoam.applications.multiphase_euler_foam_2 import luo_breakage_rate
        rate = luo_breakage_rate(d_i=1e-3, epsilon=0.1, sigma=0.07,
                                  rho_c=1.225, rho_d=1000.0)
        assert rate >= 0.0

    def test_increases_with_dissipation(self):
        from pyfoam.applications.multiphase_euler_foam_2 import luo_breakage_rate
        rate_low = luo_breakage_rate(d_i=1e-3, epsilon=0.01, sigma=0.07,
                                      rho_c=1.225, rho_d=1000.0)
        rate_high = luo_breakage_rate(d_i=1e-3, epsilon=0.1, sigma=0.07,
                                       rho_c=1.225, rho_d=1000.0)
        assert rate_high >= rate_low


class TestPrinceBlanchCoalescence:
    """Tests for Prince-Blanch coalescence kernel."""

    def test_zero_for_zero_sigma(self):
        from pyfoam.applications.multiphase_euler_foam_2 import prince_blanch_coalescence_rate
        rate = prince_blanch_coalescence_rate(
            d_i=1e-3, d_j=1e-3, epsilon=0.01, sigma=0.0, rho_c=1.225,
        )
        assert rate == 0.0

    def test_positive_for_physical_params(self):
        from pyfoam.applications.multiphase_euler_foam_2 import prince_blanch_coalescence_rate
        rate = prince_blanch_coalescence_rate(
            d_i=1e-3, d_j=1e-3, epsilon=0.1, sigma=0.07, rho_c=1000.0,
        )
        assert rate >= 0.0


class TestMultiphaseEulerFoam2Import:
    """Import tests."""

    def test_imports(self):
        from pyfoam.applications.multiphase_euler_foam_2 import MultiphaseEulerFoam2
        assert MultiphaseEulerFoam2 is not None

    def test_exports_in_all(self):
        from pyfoam.applications import MultiphaseEulerFoam2
        assert MultiphaseEulerFoam2 is not None
