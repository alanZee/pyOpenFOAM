"""Tests for turbulence wall damping models (Brackbill)."""

import pytest
import torch

from pyfoam.multiphase.turbulence_wall_damping import (
    TurbulenceWallDampingModel,
    BrackbillDamping,
)


class TestBrackbillDamping:
    """Brackbill 近壁湍流阻尼模型测试。"""

    def test_registration(self):
        """brackbillDamping 已注册到 RTS。"""
        assert "brackbillDamping" in TurbulenceWallDampingModel.available_types()

    def test_factory_creation(self):
        """工厂方法创建模型。"""
        model = TurbulenceWallDampingModel.create("brackbillDamping")
        assert isinstance(model, BrackbillDamping)

    def test_default_params(self):
        """默认参数：damping_coeff=0.5, A_wall=25.0。"""
        model = BrackbillDamping()
        assert model.damping_coeff == 0.5
        assert model.A_wall == 25.0
        assert model.alpha_min == 0.01
        assert model.alpha_max == 0.99

    def test_custom_params(self):
        """自定义参数正确存储。"""
        model = BrackbillDamping(
            damping_coeff=0.8, A_wall=30.0, alpha_min=0.05, alpha_max=0.95,
        )
        assert model.damping_coeff == 0.8
        assert model.A_wall == 30.0
        assert model.alpha_min == 0.05
        assert model.alpha_max == 0.95

    def test_interface_indicator_pure_phases(self):
        """纯相 (alpha=0 或 1) 的界面指示子为零。"""
        model = BrackbillDamping()
        alpha = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        f = model.compute_interface_indicator(alpha)
        assert torch.allclose(f, torch.zeros(4, dtype=torch.float64), atol=1e-10)

    def test_interface_indicator_at_interface(self):
        """alpha=0.5 时界面指示子最大 (=1.0)。"""
        model = BrackbillDamping()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        f = model.compute_interface_indicator(alpha)
        assert torch.allclose(f, torch.tensor([1.0], dtype=torch.float64), atol=1e-10)

    def test_damp_k_pure_phases_unchanged(self):
        """纯相的 k 不受影响。"""
        model = BrackbillDamping(damping_coeff=0.5)
        alpha = torch.tensor([0.0, 1.0], dtype=torch.float64)
        k = torch.tensor([100.0, 100.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert torch.allclose(k_damped, k, atol=1e-10)

    def test_damp_k_at_interface_without_y_plus(self):
        """无 y+ 时 (f_wall=0)，界面处阻尼 = damping_coeff * f_interface。

        alpha=0.5, f_interface=1.0, f_wall=0
        total = 0.5 * 1.0 * 1.0 = 0.5
        k_damped = k * (1 - 0.5) = 50
        """
        model = BrackbillDamping(damping_coeff=0.5)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([100.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped[0].item() == pytest.approx(50.0)

    def test_damp_k_at_wall_with_y_plus(self):
        """在壁面 (y+=0)，阻尼最大。f_wall = 1 - exp(0) = 0。total = coeff * f_int * 1。

        与无 y+ 情况相同。
        """
        model = BrackbillDamping(damping_coeff=0.5)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([100.0], dtype=torch.float64)
        y_plus = torch.tensor([0.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, y_plus=y_plus)
        assert k_damped[0].item() == pytest.approx(50.0)

    def test_damp_k_far_from_wall(self):
        """远离壁面 (y+ 很大)，f_wall~1，(1-f_wall)~0，阻尼应很小。

        y+=1000, A_wall=25: f_wall = 1 - exp(-40) ≈ 1
        total ≈ damping_coeff * f_interface * 0 = 0
        """
        model = BrackbillDamping(damping_coeff=0.5, A_wall=25.0)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([100.0], dtype=torch.float64)
        y_plus = torch.tensor([1000.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k, y_plus=y_plus)
        # 几乎无阻尼
        assert k_damped[0].item() > 99.0

    def test_damp_epsilon_reduces_at_interface(self):
        """epsilon 在界面处被阻尼。"""
        model = BrackbillDamping(damping_coeff=0.5)
        alpha = torch.tensor([0.5, 0.0], dtype=torch.float64)
        eps = torch.tensor([50.0, 50.0], dtype=torch.float64)
        eps_damped = model.damp_epsilon(alpha, eps)
        assert eps_damped[0].item() < 50.0
        assert eps_damped[1].item() == pytest.approx(50.0)

    def test_damp_omega(self):
        """omega 在界面处被阻尼。"""
        model = BrackbillDamping(damping_coeff=0.5)
        alpha = torch.tensor([0.5, 0.0], dtype=torch.float64)
        omega = torch.tensor([200.0, 200.0], dtype=torch.float64)
        omega_damped = model.damp_omega(alpha, omega)
        assert omega_damped[0].item() < 200.0
        assert omega_damped[1].item() == pytest.approx(200.0)

    def test_zero_damping_coeff_no_effect(self):
        """damping_coeff=0 时无阻尼。"""
        model = BrackbillDamping(damping_coeff=0.0)
        alpha = torch.tensor([0.5, 0.3, 0.7], dtype=torch.float64)
        k = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert torch.allclose(k_damped, k, atol=1e-10)

    def test_higher_damping_coeff_more_reduction(self):
        """更高阻尼系数产生更大阻尼。"""
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([100.0], dtype=torch.float64)

        model_low = BrackbillDamping(damping_coeff=0.1)
        model_high = BrackbillDamping(damping_coeff=0.9)

        k_low = model_low.damp_k(alpha, k)
        k_high = model_high.damp_k(alpha, k)

        assert k_low > k_high

    def test_wall_damping_increases_near_wall(self):
        """近壁 (小 y+) 比远壁 (大 y+) 有更大阻尼。"""
        model = BrackbillDamping(damping_coeff=0.5, A_wall=25.0)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([100.0], dtype=torch.float64)

        y_near = torch.tensor([1.0], dtype=torch.float64)
        y_far = torch.tensor([500.0], dtype=torch.float64)

        k_near = model.damp_k(alpha, k, y_plus=y_near)
        k_far = model.damp_k(alpha, k, y_plus=y_far)

        # 近壁阻尼更大
        assert k_near < k_far

    def test_outside_alpha_threshold_no_damping(self):
        """alpha 超出 [alpha_min, alpha_max] 时无阻尼。"""
        model = BrackbillDamping(damping_coeff=0.5, alpha_min=0.1, alpha_max=0.9)
        alpha = torch.tensor([0.001, 0.999], dtype=torch.float64)
        k = torch.tensor([100.0, 100.0], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert torch.allclose(k_damped, k, atol=1e-10)

    def test_batch_processing(self):
        """批量处理正确。"""
        model = BrackbillDamping(damping_coeff=0.5)
        n = 100
        alpha = torch.rand(n, dtype=torch.float64)
        k = torch.ones(n, dtype=torch.float64) * 10.0
        y_plus = torch.rand(n, dtype=torch.float64) * 100.0
        k_damped = model.damp_k(alpha, k, y_plus=y_plus)
        assert k_damped.shape == (n,)
        # k_damped <= k 总是成立
        assert (k_damped <= k + 1e-10).all()

    def test_batch_processing_epsilon(self):
        """epsilon 批量处理。"""
        model = BrackbillDamping(damping_coeff=0.5)
        n = 50
        alpha = torch.rand(n, dtype=torch.float64)
        eps = torch.ones(n, dtype=torch.float64) * 5.0
        eps_damped = model.damp_epsilon(alpha, eps)
        assert eps_damped.shape == (n,)
        assert (eps_damped <= eps + 1e-10).all()

    def test_damped_values_non_negative(self):
        """阻尼后值非负。"""
        model = BrackbillDamping(damping_coeff=1.0)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        k = torch.tensor([1e-6], dtype=torch.float64)
        k_damped = model.damp_k(alpha, k)
        assert k_damped[0].item() >= 0.0
