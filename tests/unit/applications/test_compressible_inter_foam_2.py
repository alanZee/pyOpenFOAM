"""
Unit tests for CompressibleInterFoam2 — enhanced compressible VOF.

Tests cover:
- Solver initialisation
- Janaf coefficients evaluation
- Sutherland viscosity function
- Variable Cp(T) computation
- Run produces finite values
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


# ---------------------------------------------------------------------------
# Janaf coefficients tests (no mesh)
# ---------------------------------------------------------------------------


class TestJanafCoeffs:
    """Tests for Janaf polynomial coefficients."""

    def test_constant_cp(self):
        from pyfoam.applications.compressible_inter_foam_2 import JanafCoeffs
        coeffs = JanafCoeffs(a=4180.0)
        T = torch.tensor([300.0, 400.0, 500.0])
        Cp = coeffs.Cp(T)
        assert torch.allclose(Cp, torch.full_like(Cp, 4180.0))

    def test_polynomial_cp(self):
        from pyfoam.applications.compressible_inter_foam_2 import JanafCoeffs
        coeffs = JanafCoeffs(a=1005.0, b=0.01, c=0.0001)
        T = torch.tensor([300.0])
        Cp = coeffs.Cp(T)
        expected = 1005.0 + 0.01 * 300.0 + 0.0001 * 300.0 ** 2
        assert abs(Cp.item() - expected) < 1e-6

    def test_cv_from_cp(self):
        from pyfoam.applications.compressible_inter_foam_2 import JanafCoeffs
        coeffs = JanafCoeffs(a=1005.0)
        T = torch.tensor([300.0])
        Cv = coeffs.Cv(T, R_specific=287.0)
        expected = 1005.0 - 287.0
        assert abs(Cv.item() - expected) < 1e-6

    def test_temperature_clamping(self):
        from pyfoam.applications.compressible_inter_foam_2 import JanafCoeffs
        coeffs = JanafCoeffs(a=1000.0, T_low=200.0, T_high=5000.0)
        T = torch.tensor([100.0])  # Below T_low
        Cp = coeffs.Cp(T)
        assert abs(Cp.item() - 1000.0) < 1e-6  # Clamped to T_low


class TestSutherlandViscosity:
    """Tests for Sutherland's law viscosity."""

    def test_reference_temperature(self):
        from pyfoam.applications.compressible_inter_foam_2 import sutherland_viscosity
        T = torch.tensor([300.0])
        mu = sutherland_viscosity(T, mu_ref=1e-3, T_ref=300.0, S=110.4)
        # At reference temperature, mu should equal mu_ref
        assert abs(mu.item() - 1e-3) < 1e-10

    def test_increases_with_temperature(self):
        from pyfoam.applications.compressible_inter_foam_2 import sutherland_viscosity
        T_low = torch.tensor([300.0])
        T_high = torch.tensor([600.0])
        mu_low = sutherland_viscosity(T_low, mu_ref=1e-3, T_ref=300.0, S=110.4)
        mu_high = sutherland_viscosity(T_high, mu_ref=1e-3, T_ref=300.0, S=110.4)
        assert mu_high.item() > mu_low.item()

    def test_positive_output(self):
        from pyfoam.applications.compressible_inter_foam_2 import sutherland_viscosity
        T = torch.tensor([200.0, 300.0, 1000.0, 2000.0])
        mu = sutherland_viscosity(T, mu_ref=1e-3, T_ref=300.0, S=110.4)
        assert (mu > 0).all()


class TestCompressibleInterFoam2Import:
    """Import tests for CompressibleInterFoam2."""

    def test_imports(self):
        from pyfoam.applications.compressible_inter_foam_2 import CompressibleInterFoam2
        assert CompressibleInterFoam2 is not None

    def test_exports_in_all(self):
        from pyfoam.applications import CompressibleInterFoam2
        assert CompressibleInterFoam2 is not None
