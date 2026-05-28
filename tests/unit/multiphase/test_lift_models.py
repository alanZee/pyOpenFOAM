"""Tests for lift model ABC hierarchy."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.multiphase.lift_models import (
    LiftModel,
    TomiyamaLift,
    SaffmanLift,
)


# ============================================================================
# LiftModel ABC
# ============================================================================


class TestLiftModelABC:
    """LiftModel 抽象基类测试。"""

    def test_rts_registry_contains_models(self):
        """两个升力模型已注册。"""
        types = LiftModel.available_types()
        assert "tomiyama" in types
        assert "saffman" in types

    def test_factory_create_tomiyama(self):
        model = LiftModel.create(
            "tomiyama", d=1e-3, rho_c=998.0, rho_d=1.225, sigma=0.072,
        )
        assert isinstance(model, TomiyamaLift)

    def test_factory_create_saffman(self):
        model = LiftModel.create(
            "saffman", d=1e-3, rho_c=998.0, mu_c=1.002e-3,
        )
        assert isinstance(model, SaffmanLift)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown lift model"):
            LiftModel.create("nonexistent", d=1e-3, rho_c=998.0)

    def test_is_abstract(self):
        """LiftModel 不能直接实例化。"""
        with pytest.raises(TypeError):
            LiftModel()


# ============================================================================
# TomiyamaLift (from lift_models)
# ============================================================================


class TestTomiyamaLift:
    """Tomiyama 升力模型测试。"""

    def test_init(self):
        lift = TomiyamaLift(d=1e-3, rho_c=998.0, rho_d=1.225, sigma=0.072)
        assert lift.d == 1e-3
        assert lift.rho_c == 998.0
        assert lift.rho_d == 1.225
        assert lift.sigma == 0.072

    def test_eotvos_number(self):
        """Eo = g * |rho_c - rho_d| * d^2 / sigma。"""
        lift = TomiyamaLift(d=1e-3, rho_c=998.0, rho_d=1.225, sigma=0.072)
        g = 9.81
        Eo_expected = g * abs(998.0 - 1.225) * (1e-3) ** 2 / 0.072
        assert lift.eotvos_number == pytest.approx(Eo_expected, rel=1e-10)

    def test_lift_coefficient_small_bubble(self):
        """小气泡 (Eo < 4) 的 C_L = 0.12。"""
        # d = 1e-4 m → Eo ≈ 0.0135 < 4 → C_L = 0.12
        lift = TomiyamaLift(d=1e-4, rho_c=998.0, rho_d=1.225, sigma=0.072)
        C_L = lift._lift_coefficient()
        assert C_L == pytest.approx(0.12, rel=1e-10)

    def test_lift_coefficient_large_bubble(self):
        """大气泡 (Eo >= 10) 的 C_L = 0.12 * 0.3 = 0.036。"""
        # d = 0.01 m → Eo ≈ 13.5 > 10 → f(Eo) = 0.3
        lift = TomiyamaLift(d=0.01, rho_c=998.0, rho_d=1.225, sigma=0.072)
        C_L = lift._lift_coefficient()
        assert C_L == pytest.approx(0.12 * 0.3, rel=1e-2)

    def test_compute_returns_finite(self):
        lift = TomiyamaLift(d=1e-3, rho_c=998.0, rho_d=1.225, sigma=0.072)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        U_rel = torch.randn(10, 3, dtype=CFD_DTYPE) * 0.1
        vorticity = torch.randn(10, 3, dtype=CFD_DTYPE) * 0.1
        F_L = lift.compute(alpha, U_rel, vorticity)
        assert F_L.shape == (10, 3)
        assert torch.isfinite(F_L).all()

    def test_force_direction(self):
        """升力方向垂直于相对速度和涡量。"""
        lift = TomiyamaLift(d=1e-4, rho_c=998.0, rho_d=1.225, sigma=0.072)
        alpha = torch.full((1,), 0.1, dtype=CFD_DTYPE)
        # U_rel = +x, vorticity = +z → F_L ~ -y (右手定则: x×z = -y)
        U_rel = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)
        vorticity = torch.tensor([[0.0, 0.0, 1.0]], dtype=CFD_DTYPE)
        F_L = lift.compute(alpha, U_rel, vorticity)
        assert F_L[0, 0].abs().item() < 1e-10  # 无 x 分量
        assert F_L[0, 1].item() < 0  # -y
        assert F_L[0, 2].abs().item() < 1e-10  # 无 z 分量

    def test_scales_with_alpha(self):
        """升力与 alpha 成正比。"""
        lift = TomiyamaLift(d=1e-4, rho_c=998.0, rho_d=1.225, sigma=0.072)
        U_rel = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)
        vorticity = torch.tensor([[0.0, 0.0, 1.0]], dtype=CFD_DTYPE)
        alpha1 = torch.tensor([0.1], dtype=CFD_DTYPE)
        alpha2 = torch.tensor([0.3], dtype=CFD_DTYPE)
        F1 = lift.compute(alpha1, U_rel, vorticity)
        F2 = lift.compute(alpha2, U_rel, vorticity)
        assert torch.allclose(F2, F1 * 3.0, rtol=1e-10)

    def test_zero_vorticity_zero_force(self):
        """零涡量 → 零升力。"""
        lift = TomiyamaLift(d=1e-4, rho_c=998.0, rho_d=1.225, sigma=0.072)
        alpha = torch.full((5,), 0.1, dtype=CFD_DTYPE)
        U_rel = torch.randn(5, 3, dtype=CFD_DTYPE)
        vorticity = torch.zeros(5, 3, dtype=CFD_DTYPE)
        F_L = lift.compute(alpha, U_rel, vorticity)
        assert torch.allclose(F_L, torch.zeros(5, 3, dtype=CFD_DTYPE))


# ============================================================================
# SaffmanLift (from lift_models)
# ============================================================================


class TestSaffmanLift:
    """Saffman 升力模型测试。"""

    def test_init(self):
        lift = SaffmanLift(d=1e-3, rho_c=998.0, mu_c=1.002e-3)
        assert lift.d == 1e-3
        assert lift.rho_c == 998.0
        assert lift.mu_c == 1.002e-3
        assert lift.C_S == 1.615  # 默认值

    def test_custom_coefficient(self):
        lift = SaffmanLift(d=1e-3, rho_c=998.0, mu_c=1.002e-3, C_S=2.0)
        assert lift.C_S == 2.0

    def test_nu_c(self):
        """nu_c = mu_c / rho_c。"""
        lift = SaffmanLift(d=1e-3, rho_c=1000.0, mu_c=1.0)
        assert lift.nu_c == pytest.approx(0.001)

    def test_compute_returns_finite(self):
        lift = SaffmanLift(d=1e-3, rho_c=998.0, mu_c=1.002e-3)
        alpha = torch.full((10,), 0.1, dtype=CFD_DTYPE)
        U_rel = torch.randn(10, 3, dtype=CFD_DTYPE) * 0.1
        vorticity = torch.randn(10, 3, dtype=CFD_DTYPE) * 10.0
        F_L = lift.compute(alpha, U_rel, vorticity)
        assert F_L.shape == (10, 3)
        assert torch.isfinite(F_L).all()

    def test_compute_with_scalar_vorticity(self):
        """支持标量涡量输入。"""
        lift = SaffmanLift(d=1e-3, rho_c=998.0, mu_c=1.002e-3)
        alpha = torch.full((5,), 0.1, dtype=CFD_DTYPE)
        U_rel = torch.randn(5, 3, dtype=CFD_DTYPE) * 0.1
        vorticity = torch.full((5,), 10.0, dtype=CFD_DTYPE)
        F_L = lift.compute(alpha, U_rel, vorticity)
        assert F_L.shape == (5, 3)
        assert torch.isfinite(F_L).all()

    def test_compute_with_strain_magnitude(self):
        """显式提供 strain_magnitude 参数。"""
        lift = SaffmanLift(d=1e-3, rho_c=998.0, mu_c=1.002e-3)
        alpha = torch.full((5,), 0.1, dtype=CFD_DTYPE)
        U_rel = torch.randn(5, 3, dtype=CFD_DTYPE) * 0.1
        vorticity = torch.randn(5, 3, dtype=CFD_DTYPE) * 10.0
        strain = torch.full((5,), 50.0, dtype=CFD_DTYPE)
        F_L = lift.compute(alpha, U_rel, vorticity, strain_magnitude=strain)
        assert F_L.shape == (5, 3)
        assert torch.isfinite(F_L).all()

    def test_force_direction_perpendicular(self):
        """升力方向垂直于相对速度和涡量。"""
        lift = SaffmanLift(d=1e-3, rho_c=998.0, mu_c=1.002e-3)
        alpha = torch.tensor([0.1], dtype=CFD_DTYPE)
        # U_rel = +x, vorticity = +z → F_L ~ +y
        U_rel = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)
        vorticity = torch.tensor([[0.0, 0.0, 10.0]], dtype=CFD_DTYPE)
        F_L = lift.compute(alpha, U_rel, vorticity)
        # x 和 z 分量应接近零
        assert F_L[0, 0].abs().item() < 1e-10
        assert F_L[0, 2].abs().item() < 1e-10
        # y 分量非零
        assert F_L[0, 1].abs().item() > 0

    def test_scales_with_alpha(self):
        """升力与 alpha 成正比。"""
        lift = SaffmanLift(d=1e-3, rho_c=998.0, mu_c=1.002e-3)
        U_rel = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)
        vorticity = torch.tensor([[0.0, 0.0, 10.0]], dtype=CFD_DTYPE)
        alpha1 = torch.tensor([0.1], dtype=CFD_DTYPE)
        alpha2 = torch.tensor([0.4], dtype=CFD_DTYPE)
        F1 = lift.compute(alpha1, U_rel, vorticity)
        F2 = lift.compute(alpha2, U_rel, vorticity)
        assert torch.allclose(F2, F1 * 4.0, rtol=1e-10)
