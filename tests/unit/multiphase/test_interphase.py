"""
Unit tests for interphase force models.

Tests verify drag, lift, and virtual mass models.
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE


class TestSchillerNaumannDrag:
    """Tests for Schiller-Naumann drag model."""

    def test_init(self):
        from pyfoam.multiphase.interphase_models import SchillerNaumannDrag

        drag = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        assert drag.d == 1e-3
        assert drag.rho_c == 1.225

    def test_compute_returns_positive(self):
        """Drag coefficient K should be positive."""
        from pyfoam.multiphase.interphase_models import SchillerNaumannDrag

        drag = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)
        U_rel = torch.full((10,), 0.1, dtype=CFD_DTYPE)

        K = drag.compute(alpha, U_rel)
        assert (K >= 0).all()
        assert torch.isfinite(K).all()

    def test_drag_increases_with_velocity(self):
        """Higher relative velocity → higher drag."""
        from pyfoam.multiphase.interphase_models import SchillerNaumannDrag

        drag = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)

        K_low = drag.compute(alpha, torch.full((10,), 0.01, dtype=CFD_DTYPE))
        K_high = drag.compute(alpha, torch.full((10,), 1.0, dtype=CFD_DTYPE))

        assert K_high.mean() > K_low.mean()

    def test_drag_increases_with_alpha(self):
        """Higher volume fraction → higher drag."""
        from pyfoam.multiphase.interphase_models import SchillerNaumannDrag

        drag = SchillerNaumannDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        U_rel = torch.full((10,), 0.1, dtype=CFD_DTYPE)

        K_low = drag.compute(torch.full((10,), 0.1, dtype=CFD_DTYPE), U_rel)
        K_high = drag.compute(torch.full((10,), 0.5, dtype=CFD_DTYPE), U_rel)

        assert K_high.mean() > K_low.mean()


class TestWenYuDrag:
    """Tests for Wen-Yu drag model."""

    def test_compute(self):
        from pyfoam.multiphase.interphase_models import WenYuDrag

        drag = WenYuDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        U_rel = torch.full((10,), 0.1, dtype=CFD_DTYPE)

        K = drag.compute(alpha, U_rel)
        assert (K >= 0).all()
        assert torch.isfinite(K).all()


class TestGidaspowDrag:
    """Tests for Gidaspow drag model."""

    def test_compute(self):
        from pyfoam.multiphase.interphase_models import GidaspowDrag

        drag = GidaspowDrag(d=1e-3, rho_c=1.225, mu_c=1.8e-5)
        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)
        U_rel = torch.full((10,), 0.1, dtype=CFD_DTYPE)

        K = drag.compute(alpha, U_rel)
        assert (K >= 0).all()
        assert torch.isfinite(K).all()


class TestTomiyamaLift:
    """Tests for Tomiyama lift model."""

    def test_init(self):
        from pyfoam.multiphase.interphase_models import TomiyamaLift

        lift = TomiyamaLift(d=1e-3, rho_c=1.225)
        assert lift.d == 1e-3

    def test_compute_returns_finite(self):
        from pyfoam.multiphase.interphase_models import TomiyamaLift

        lift = TomiyamaLift(d=1e-3, rho_c=1.225)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        U_rel = torch.randn(10, 3, dtype=CFD_DTYPE) * 0.1
        vorticity = torch.randn(10, 3, dtype=CFD_DTYPE) * 0.1

        F_L = lift.compute(alpha, U_rel, vorticity)
        assert F_L.shape == (10, 3)
        assert torch.isfinite(F_L).all()


class TestVirtualMassForce:
    """Tests for virtual mass force model."""

    def test_init(self):
        from pyfoam.multiphase.interphase_models import VirtualMassForce

        vm = VirtualMassForce(C_vm=0.5)
        assert vm.C_vm == 0.5

    def test_compute(self):
        from pyfoam.multiphase.interphase_models import VirtualMassForce

        vm = VirtualMassForce(C_vm=0.5)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        Ddt_rel = torch.randn(10, 3, dtype=CFD_DTYPE) * 0.01

        F_vm = vm.compute(alpha, 1.225, Ddt_rel)
        assert F_vm.shape == (10, 3)
        assert torch.isfinite(F_vm).all()
