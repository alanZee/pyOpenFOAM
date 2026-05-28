"""Tests for production limiter models."""

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.turbulence.production_limiter import (
    ProductionLimiter,
    StandardLimiter,
    KatoLimiter,
)


# ============================================================================
# ProductionLimiter ABC
# ============================================================================


class TestProductionLimiterABC:
    """ProductionLimiter 抽象基类测试。"""

    def test_rts_registry_contains_models(self):
        """两个限幅器已注册。"""
        types = ProductionLimiter.available_types()
        assert "standard" in types
        assert "kato" in types

    def test_factory_create_standard(self):
        limiter = ProductionLimiter.create("standard", C_lim=20.0)
        assert isinstance(limiter, StandardLimiter)

    def test_factory_create_kato(self):
        limiter = ProductionLimiter.create("kato", nu_t=1e-3)
        assert isinstance(limiter, KatoLimiter)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown production limiter"):
            ProductionLimiter.create("nonexistent")

    def test_is_abstract(self):
        """ProductionLimiter 不能直接实例化。"""
        with pytest.raises(TypeError):
            ProductionLimiter()


# ============================================================================
# StandardLimiter
# ============================================================================


class TestStandardLimiter:
    """标准生产限幅器测试。"""

    def test_init_default(self):
        limiter = StandardLimiter()
        assert limiter.C_lim == 20.0

    def test_init_custom(self):
        limiter = StandardLimiter(C_lim=10.0)
        assert limiter.C_lim == 10.0

    def test_limit_below_threshold(self):
        """P_k < C_lim * epsilon 时不限幅。"""
        limiter = StandardLimiter(C_lim=20.0)
        P_k = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        epsilon = torch.full((5,), 10.0, dtype=CFD_DTYPE)  # C_lim * eps = 200
        P_limited = limiter.limit(P_k, epsilon)
        assert torch.allclose(P_limited, P_k)

    def test_limit_above_threshold(self):
        """P_k > C_lim * epsilon 时截断。"""
        limiter = StandardLimiter(C_lim=20.0)
        P_k = torch.full((5,), 500.0, dtype=CFD_DTYPE)
        epsilon = torch.full((5,), 10.0, dtype=CFD_DTYPE)  # C_lim * eps = 200
        P_limited = limiter.limit(P_k, epsilon)
        assert torch.allclose(
            P_limited, torch.full((5,), 200.0, dtype=CFD_DTYPE),
        )

    def test_limit_at_threshold(self):
        """P_k = C_lim * epsilon 时恰好不限幅。"""
        limiter = StandardLimiter(C_lim=20.0)
        P_k = torch.full((5,), 200.0, dtype=CFD_DTYPE)
        epsilon = torch.full((5,), 10.0, dtype=CFD_DTYPE)
        P_limited = limiter.limit(P_k, epsilon)
        assert torch.allclose(P_limited, P_k)

    def test_limit_mixed(self):
        """混合：部分限幅、部分不限幅。"""
        limiter = StandardLimiter(C_lim=20.0)
        P_k = torch.tensor([100.0, 500.0, 50.0, 1000.0, 200.0], dtype=CFD_DTYPE)
        epsilon = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0], dtype=CFD_DTYPE)
        P_limited = limiter.limit(P_k, epsilon)
        expected = torch.tensor([100.0, 200.0, 50.0, 200.0, 200.0], dtype=CFD_DTYPE)
        assert torch.allclose(P_limited, expected)

    def test_output_shape(self):
        """输出形状与输入一致。"""
        limiter = StandardLimiter()
        P_k = torch.randn(20, dtype=CFD_DTYPE).abs()
        epsilon = torch.randn(20, dtype=CFD_DTYPE).abs() + 0.01
        P_limited = limiter.limit(P_k, epsilon)
        assert P_limited.shape == P_k.shape

    def test_output_finite(self):
        """输出始终有限。"""
        limiter = StandardLimiter()
        P_k = torch.randn(10, dtype=CFD_DTYPE) * 1000
        epsilon = torch.randn(10, dtype=CFD_DTYPE).abs() + 1e-10
        P_limited = limiter.limit(P_k, epsilon)
        assert torch.isfinite(P_limited).all()

    def test_custom_C_lim(self):
        """自定义 C_lim 值。"""
        limiter = StandardLimiter(C_lim=10.0)
        P_k = torch.full((3,), 150.0, dtype=CFD_DTYPE)
        epsilon = torch.full((3,), 10.0, dtype=CFD_DTYPE)
        P_limited = limiter.limit(P_k, epsilon)
        assert torch.allclose(
            P_limited, torch.full((3,), 100.0, dtype=CFD_DTYPE),
        )

    def test_scales_with_epsilon(self):
        """epsilon 增大时限幅阈值增大。"""
        limiter = StandardLimiter(C_lim=20.0)
        P_k = torch.full((1,), 500.0, dtype=CFD_DTYPE)
        eps_low = torch.full((1,), 10.0, dtype=CFD_DTYPE)
        eps_high = torch.full((1,), 100.0, dtype=CFD_DTYPE)
        P_low = limiter.limit(P_k, eps_low)
        P_high = limiter.limit(P_k, eps_high)
        assert P_low.item() < P_high.item()


# ============================================================================
# KatoLimiter
# ============================================================================


class TestKatoLimiter:
    """Kato-Launder 限幅器测试。"""

    def test_init_default(self):
        limiter = KatoLimiter()
        assert limiter.nu_t is None

    def test_init_with_nu_t(self):
        limiter = KatoLimiter(nu_t=1e-3)
        assert limiter.nu_t == 1e-3

    def test_limit_with_nu_t(self):
        """提供 nu_t 时限幅到 Kato-Launder 生产。"""
        nu_t = 1e-2
        limiter = KatoLimiter(nu_t=nu_t)
        # P_k = 2 * nu_t * |S|^2，设 |S| = 100 → P_k = 2 * 0.01 * 10000 = 200
        P_k = torch.full((5,), 200.0, dtype=CFD_DTYPE)
        epsilon = torch.full((5,), 50.0, dtype=CFD_DTYPE)
        P_limited = limiter.limit(P_k, epsilon)
        # Kato: P_k_KL = nu_t * |S| * |Omega|
        # |S| = sqrt(P_k / (2*nu_t)) = sqrt(200/0.02) = 100
        # |Omega| = sqrt(eps/nu_t) = sqrt(50/0.01) = ~70.71
        # P_k_KL = 0.01 * 100 * 70.71 ≈ 70.71
        assert P_limited.shape == (5,)
        assert torch.isfinite(P_limited).all()
        # Kato 生产应小于原 P_k
        assert (P_limited < P_k).all()

    def test_limit_without_nu_t_fallback(self):
        """无 nu_t 时回退到标准限幅器。"""
        limiter = KatoLimiter(nu_t=None)
        P_k = torch.full((5,), 500.0, dtype=CFD_DTYPE)
        epsilon = torch.full((5,), 10.0, dtype=CFD_DTYPE)
        P_limited = limiter.limit(P_k, epsilon)
        # 回退: min(P_k, 20 * epsilon) = min(500, 200) = 200
        assert torch.allclose(
            P_limited, torch.full((5,), 200.0, dtype=CFD_DTYPE),
        )

    def test_limit_with_strain_static(self):
        """静态方法 limit_with_strain 直接计算 Kato-Launder 生产。"""
        nu_t = torch.full((5,), 0.01, dtype=CFD_DTYPE)
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        omega = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        P_k = torch.full((5,), 200.0, dtype=CFD_DTYPE)
        P_kl = KatoLimiter.limit_with_strain(P_k, strain, omega, nu_t)
        # P_k_KL = 2 * 0.01 * 100 * 100 = 200
        assert torch.allclose(
            P_kl, torch.full((5,), 200.0, dtype=CFD_DTYPE),
        )

    def test_limit_with_strain_reduces_at_stagnation(self):
        """在驻点（|Omega| << |S|）时 Kato-Launder 显著减小生产。"""
        nu_t = torch.full((5,), 0.01, dtype=CFD_DTYPE)
        # 高应变率，低涡量（驻点特征）
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        omega = torch.full((5,), 1.0, dtype=CFD_DTYPE)
        P_k = torch.full((5,), 200.0, dtype=CFD_DTYPE)  # = 2*nu_t*|S|^2
        P_kl = KatoLimiter.limit_with_strain(P_k, strain, omega, nu_t)
        # P_k_KL = 2 * 0.01 * 100 * 1 = 2.0 << 200
        assert torch.allclose(
            P_kl, torch.full((5,), 2.0, dtype=CFD_DTYPE),
        )

    def test_limit_with_strain_equal_in_shear(self):
        """在剪切层（|S| ≈ |Omega|）时 Kato-Launder ≈ 原生产。"""
        nu_t = torch.full((5,), 0.01, dtype=CFD_DTYPE)
        strain = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        omega = torch.full((5,), 100.0, dtype=CFD_DTYPE)
        # 原生产: P_k = 2 * nu_t * |S|^2 = 200
        P_k = torch.full((5,), 200.0, dtype=CFD_DTYPE)
        P_kl = KatoLimiter.limit_with_strain(P_k, strain, omega, nu_t)
        # P_k_KL = 2 * 0.01 * 100 * 100 = 200 (相同)
        assert torch.allclose(P_kl, P_k)

    def test_output_finite(self):
        """输出始终有限。"""
        limiter = KatoLimiter(nu_t=1e-3)
        P_k = torch.randn(10, dtype=CFD_DTYPE).abs() * 100
        epsilon = torch.randn(10, dtype=CFD_DTYPE).abs() + 1e-10
        P_limited = limiter.limit(P_k, epsilon)
        assert torch.isfinite(P_limited).all()
