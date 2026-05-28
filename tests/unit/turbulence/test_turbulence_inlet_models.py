"""Tests for turbulence inlet models."""

import pytest
import torch

from pyfoam.turbulence.turbulence_inlet_models import (
    TurbulenceInletModel,
    FixedTurbulenceInlet,
    MappedTurbulenceInlet,
)


class TestFixedTurbulenceInlet:
    """FixedTurbulenceInlet 测试。"""

    def test_registration(self):
        """fixedTurbulenceInlet 已注册到 RTS。"""
        assert "fixedTurbulenceInlet" in TurbulenceInletModel.available_types()

    def test_factory_creation(self):
        model = TurbulenceInletModel.create("fixedTurbulenceInlet")
        assert isinstance(model, FixedTurbulenceInlet)

    def test_default_params(self):
        """默认参数：k=0.01, epsilon=0.001, omega=None。"""
        model = FixedTurbulenceInlet()
        assert model.k_value == 0.01
        assert model.epsilon_value == 0.001
        assert model.omega_value is None

    def test_custom_params(self):
        model = FixedTurbulenceInlet(k=0.05, epsilon=0.01, omega=100.0)
        assert model.k_value == 0.05
        assert model.epsilon_value == 0.01
        assert model.omega_value == 100.0

    def test_intensity_based_k(self):
        """强度法计算 k: k = 1.5 * (I * U_ref)^2。"""
        model = FixedTurbulenceInlet(intensity=0.05, U_ref=10.0)
        expected_k = 1.5 * (0.05 * 10.0) ** 2  # = 0.375
        assert model.k_value == pytest.approx(expected_k)

    def test_length_scale_based_epsilon(self):
        """长度尺度法计算 epsilon。"""
        model = FixedTurbulenceInlet(k=1.0, length_scale=0.1)
        C_mu = 0.09
        expected_eps = C_mu ** 0.75 * 1.0 ** 1.5 / 0.1
        assert model.epsilon_value == pytest.approx(expected_eps)

    def test_compute_k_uniform(self):
        model = FixedTurbulenceInlet(k=0.05)
        k = model.compute_k(n_faces=100)
        assert k.shape == (100,)
        assert torch.allclose(k, torch.full((100,), 0.05, dtype=k.dtype))

    def test_compute_epsilon_uniform(self):
        model = FixedTurbulenceInlet(epsilon=0.01)
        eps = model.compute_epsilon(n_faces=50)
        assert eps.shape == (50,)
        assert torch.allclose(eps, torch.full((50,), 0.01, dtype=eps.dtype))

    def test_compute_omega_auto_from_k_epsilon(self):
        """omega 从 k 和 epsilon 自动计算: omega = epsilon / (C_mu * k)。"""
        model = FixedTurbulenceInlet(k=0.01, epsilon=0.001)
        omega = model.compute_omega(n_faces=10)
        expected = 0.001 / (0.09 * 0.01)
        assert omega.shape == (10,)
        assert torch.allclose(omega, torch.full((10,), expected, dtype=omega.dtype), rtol=1e-4)

    def test_compute_omega_explicit(self):
        """显式指定 omega。"""
        model = FixedTurbulenceInlet(k=0.01, epsilon=0.001, omega=50.0)
        omega = model.compute_omega(n_faces=5)
        assert torch.allclose(omega, torch.full((5,), 50.0, dtype=omega.dtype))

    def test_compute_omega_positive(self):
        """omega 始终为正。"""
        model = FixedTurbulenceInlet(k=1e-6, epsilon=1e-8)
        omega = model.compute_omega(n_faces=1)
        assert omega[0].item() > 0.0

    def test_zero_faces(self):
        """空面列表正确处理。"""
        model = FixedTurbulenceInlet(k=0.01)
        k = model.compute_k(n_faces=0)
        assert k.shape == (0,)


class TestMappedTurbulenceInlet:
    """MappedTurbulenceInlet 测试。"""

    def test_registration(self):
        """mappedTurbulenceInlet 已注册到 RTS。"""
        assert "mappedTurbulenceInlet" in TurbulenceInletModel.available_types()

    def test_factory_creation(self):
        model = TurbulenceInletModel.create("mappedTurbulenceInlet")
        assert isinstance(model, MappedTurbulenceInlet)

    def test_default_params(self):
        model = MappedTurbulenceInlet()
        assert model.scale_k == 1.0
        assert model.scale_epsilon == 1.0

    def test_custom_params(self):
        model = MappedTurbulenceInlet(scale_k=2.0, scale_epsilon=0.5)
        assert model.scale_k == 2.0
        assert model.scale_epsilon == 0.5

    def test_no_reference_returns_zeros(self):
        """无参考数据时返回零。"""
        model = MappedTurbulenceInlet()
        k = model.compute_k(n_faces=10)
        eps = model.compute_epsilon(n_faces=10)
        assert torch.allclose(k, torch.zeros(10, dtype=k.dtype))
        assert torch.allclose(eps, torch.zeros(10, dtype=eps.dtype))

    def test_scalar_reference(self):
        """标量参考数据广播到所有面。"""
        model = MappedTurbulenceInlet()
        k_ref = torch.tensor(0.05, dtype=torch.float64)
        eps_ref = torch.tensor(0.005, dtype=torch.float64)
        model.set_reference(k_ref, eps_ref)

        k = model.compute_k(n_faces=10)
        eps = model.compute_epsilon(n_faces=10)

        assert k.shape == (10,)
        assert torch.allclose(k, torch.full((10,), 0.05, dtype=k.dtype))
        assert torch.allclose(eps, torch.full((10,), 0.005, dtype=eps.dtype))

    def test_matching_length_reference(self):
        """参考数据长度匹配面数时直接使用。"""
        model = MappedTurbulenceInlet()
        k_ref = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        eps_ref = torch.tensor([0.001, 0.002, 0.003], dtype=torch.float64)
        model.set_reference(k_ref, eps_ref)

        k = model.compute_k(n_faces=3)
        eps = model.compute_epsilon(n_faces=3)

        assert torch.allclose(k, k_ref)
        assert torch.allclose(eps, eps_ref)

    def test_resampling(self):
        """不同长度参考数据进行线性重采样。"""
        model = MappedTurbulenceInlet()
        k_ref = torch.tensor([0.0, 1.0], dtype=torch.float64)
        eps_ref = torch.tensor([0.0, 1.0], dtype=torch.float64)
        model.set_reference(k_ref, eps_ref)

        # 重采样到 3 个面：[0.0, 0.5, 1.0]
        k = model.compute_k(n_faces=3)
        assert k[0].item() == pytest.approx(0.0)
        assert k[1].item() == pytest.approx(0.5)
        assert k[2].item() == pytest.approx(1.0)

    def test_scale_factors(self):
        """缩放因子正确应用。"""
        model = MappedTurbulenceInlet(scale_k=2.0, scale_epsilon=0.5)
        k_ref = torch.tensor([0.01, 0.02], dtype=torch.float64)
        eps_ref = torch.tensor([0.002, 0.004], dtype=torch.float64)
        model.set_reference(k_ref, eps_ref)

        k = model.compute_k(n_faces=2)
        eps = model.compute_epsilon(n_faces=2)

        assert torch.allclose(k, torch.tensor([0.02, 0.04], dtype=k.dtype))
        assert torch.allclose(eps, torch.tensor([0.001, 0.002], dtype=eps.dtype))

    def test_compute_omega_from_reference(self):
        """omega 从 k 和 epsilon 参考数据计算。"""
        model = MappedTurbulenceInlet()
        k_ref = torch.tensor([0.01], dtype=torch.float64)
        eps_ref = torch.tensor([0.001], dtype=torch.float64)
        model.set_reference(k_ref, eps_ref)

        omega = model.compute_omega(n_faces=1)
        expected = 0.001 / (0.09 * 0.01)
        assert omega[0].item() == pytest.approx(expected, rel=1e-4)

    def test_compute_omega_explicit_reference(self):
        """显式 omega 参考数据。"""
        model = MappedTurbulenceInlet()
        k_ref = torch.tensor([0.01], dtype=torch.float64)
        eps_ref = torch.tensor([0.001], dtype=torch.float64)
        omega_ref = torch.tensor([50.0], dtype=torch.float64)
        model.set_reference(k_ref, eps_ref, omega_ref=omega_ref)

        omega = model.compute_omega(n_faces=1)
        assert omega[0].item() == pytest.approx(50.0)

    def test_output_always_non_negative(self):
        """输出值非负（clamp）。"""
        model = MappedTurbulenceInlet()
        k_ref = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        eps_ref = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        model.set_reference(k_ref, eps_ref)

        k = model.compute_k(n_faces=3)
        eps = model.compute_epsilon(n_faces=3)
        assert (k >= 0).all()
        assert (eps >= 0).all()  # eps clamped to 1e-30
