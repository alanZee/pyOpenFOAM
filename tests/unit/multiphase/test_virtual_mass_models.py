"""Tests for virtual mass force models.

Tests cover:
- VirtualMassModel ABC
- RTS registry
- Factory creation
- ConstantVirtualMass model
- LambVirtualMass model
"""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.multiphase.virtual_mass_models import (
    VirtualMassModel,
    ConstantVirtualMass,
    LambVirtualMass,
)


# ============================================================================
# VirtualMassModel ABC
# ============================================================================


class TestVirtualMassModelABC:
    """VirtualMassModel 抽象基类测试。"""

    def test_rts_registry_contains_models(self):
        types = VirtualMassModel.available_types()
        assert "constant" in types
        assert "lamb" in types

    def test_factory_create_constant(self):
        vm = VirtualMassModel.create("constant", rho_c=1000.0, C_vm=0.5)
        assert isinstance(vm, ConstantVirtualMass)

    def test_factory_create_lamb(self):
        vm = VirtualMassModel.create("lamb", rho_c=1000.0)
        assert isinstance(vm, LambVirtualMass)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown virtual mass model"):
            VirtualMassModel.create("nonexistent", rho_c=1000.0)

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            VirtualMassModel(rho_c=1000.0)


# ============================================================================
# ConstantVirtualMass
# ============================================================================


class TestConstantVirtualMass:
    """Constant virtual mass coefficient model tests."""

    def test_init_default(self):
        vm = ConstantVirtualMass(rho_c=1000.0)
        assert vm.C_vm == pytest.approx(0.5)
        assert vm.rho_c == pytest.approx(1000.0)

    def test_init_custom(self):
        vm = ConstantVirtualMass(rho_c=998.0, C_vm=1.0)
        assert vm.C_vm == pytest.approx(1.0)
        assert vm.rho_c == pytest.approx(998.0)

    def test_compute_shape(self):
        vm = ConstantVirtualMass(rho_c=1000.0, C_vm=0.5)
        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)
        DUDt_c = torch.randn(10, 3, dtype=CFD_DTYPE) * 10
        DUDt_d = torch.randn(10, 3, dtype=CFD_DTYPE) * 10
        F = vm.compute(alpha, DUDt_c, DUDt_d)
        assert F.shape == (10, 3)

    def test_compute_formula(self):
        """F = C_vm * rho_c * alpha * (DUDt_c - DUDt_d)."""
        vm = ConstantVirtualMass(rho_c=1000.0, C_vm=0.5)
        alpha = torch.full((5,), 0.3, dtype=CFD_DTYPE)
        DUDt_c = torch.ones(5, 3, dtype=CFD_DTYPE) * 10.0
        DUDt_d = torch.ones(5, 3, dtype=CFD_DTYPE) * 2.0
        F = vm.compute(alpha, DUDt_c, DUDt_d)
        expected = 0.5 * 1000.0 * 0.3 * (10.0 - 2.0)
        assert torch.allclose(F[:, 0], torch.full((5,), expected, dtype=CFD_DTYPE))

    def test_zero_acceleration_difference(self):
        """Same acceleration → zero force."""
        vm = ConstantVirtualMass(rho_c=1000.0, C_vm=0.5)
        alpha = torch.full((5,), 0.3, dtype=CFD_DTYPE)
        DUDt = torch.randn(5, 3, dtype=CFD_DTYPE)
        F = vm.compute(alpha, DUDt, DUDt)
        assert torch.allclose(F, torch.zeros(5, 3, dtype=CFD_DTYPE))

    def test_force_increases_with_alpha(self):
        vm = ConstantVirtualMass(rho_c=1000.0, C_vm=0.5)
        DUDt_c = torch.ones(5, 3, dtype=CFD_DTYPE) * 10.0
        DUDt_d = torch.zeros(5, 3, dtype=CFD_DTYPE)
        F_low = vm.compute(torch.full((5,), 0.1, dtype=CFD_DTYPE), DUDt_c, DUDt_d)
        F_high = vm.compute(torch.full((5,), 0.5, dtype=CFD_DTYPE), DUDt_c, DUDt_d)
        assert F_high.norm() > F_low.norm()

    def test_output_finite(self):
        vm = ConstantVirtualMass(rho_c=1000.0, C_vm=0.5)
        alpha = torch.rand(10, dtype=CFD_DTYPE)
        DUDt_c = torch.randn(10, 3, dtype=CFD_DTYPE) * 100
        DUDt_d = torch.randn(10, 3, dtype=CFD_DTYPE) * 100
        F = vm.compute(alpha, DUDt_c, DUDt_d)
        assert torch.isfinite(F).all()


# ============================================================================
# LambVirtualMass
# ============================================================================


class TestLambVirtualMass:
    """Lamb inviscid virtual mass model tests."""

    def test_init(self):
        vm = LambVirtualMass(rho_c=1000.0)
        assert vm.rho_c == pytest.approx(1000.0)

    def test_C_vm_always_half(self):
        """Lamb's coefficient is always 0.5."""
        vm = LambVirtualMass(rho_c=1000.0)
        assert vm.C_vm == pytest.approx(0.5)

    def test_compute_shape(self):
        vm = LambVirtualMass(rho_c=1000.0)
        alpha = torch.full((10,), 0.3, dtype=CFD_DTYPE)
        DUDt_c = torch.randn(10, 3, dtype=CFD_DTYPE)
        DUDt_d = torch.randn(10, 3, dtype=CFD_DTYPE)
        F = vm.compute(alpha, DUDt_c, DUDt_d)
        assert F.shape == (10, 3)

    def test_compute_formula(self):
        """F = 0.5 * rho_c * alpha * (DUDt_c - DUDt_d)."""
        vm = LambVirtualMass(rho_c=1000.0)
        alpha = torch.full((5,), 0.3, dtype=CFD_DTYPE)
        DUDt_c = torch.ones(5, 3, dtype=CFD_DTYPE) * 10.0
        DUDt_d = torch.ones(5, 3, dtype=CFD_DTYPE) * 2.0
        F = vm.compute(alpha, DUDt_c, DUDt_d)
        expected = 0.5 * 1000.0 * 0.3 * (10.0 - 2.0)
        assert torch.allclose(F[:, 0], torch.full((5,), expected, dtype=CFD_DTYPE))

    def test_matches_constant_with_0_5(self):
        """Lamb matches ConstantVirtualMass with C_vm=0.5."""
        lamb = LambVirtualMass(rho_c=1000.0)
        const = ConstantVirtualMass(rho_c=1000.0, C_vm=0.5)
        alpha = torch.rand(10, dtype=CFD_DTYPE)
        DUDt_c = torch.randn(10, 3, dtype=CFD_DTYPE) * 10
        DUDt_d = torch.randn(10, 3, dtype=CFD_DTYPE) * 10
        F_lamb = lamb.compute(alpha, DUDt_c, DUDt_d)
        F_const = const.compute(alpha, DUDt_c, DUDt_d)
        assert torch.allclose(F_lamb, F_const)
