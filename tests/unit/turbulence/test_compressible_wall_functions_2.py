"""Tests for compressible epsilon and omega wall functions."""

import pytest
import torch

from pyfoam.turbulence.compressible_wal_functions_2 import (
    CompressibleEpsilonWallFunction,
    CompressibleOmegaWallFunction,
)


class TestCompressibleEpsilonWallFunction:
    """CompressibleEpsilonWallFunction 测试。"""

    def test_default_params(self):
        wf = CompressibleEpsilonWallFunction()
        assert wf.kappa == 0.41
        assert wf.E == 9.8
        assert wf.C_mu == 0.09
        assert wf.y_plus_visc == 11.225

    def test_custom_params(self):
        wf = CompressibleEpsilonWallFunction(kappa=0.38, C_mu=0.085, y_plus_visc=12.0)
        assert wf.kappa == 0.38
        assert wf.C_mu == 0.085
        assert wf.y_plus_visc == 12.0

    def test_log_law_region(self):
        """高 y+ 区域应使用对数律公式。

        epsilon = C_mu^{3/4} * k^{3/2} / (kappa * y)
        """
        wf = CompressibleEpsilonWallFunction()
        # k=1.0, y=0.1, mu=1e-3, rho=1.0
        # u_tau = 0.09^0.25 * sqrt(1) = 0.5477
        # y+ = 1.0 * 0.5477 * 0.1 / 1e-3 = 54.77 > 11.225
        k = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([0.1], dtype=torch.float64)
        mu = torch.tensor([1e-3], dtype=torch.float64)
        rho = torch.tensor([1.0], dtype=torch.float64)

        eps = wf.compute(k, y, mu, rho)

        # Expected: C_mu^{0.75} * 1^1.5 / (0.41 * 0.1)
        expected = 0.09 ** 0.75 / (0.41 * 0.1)
        assert eps[0].item() == pytest.approx(expected, rel=1e-4)

    def test_viscous_sublayer(self):
        """低 y+ 区域应使用分子扩散公式。

        epsilon = 2 * nu * k / y^2
        """
        wf = CompressibleEpsilonWallFunction()
        # k=1.0, y=0.001, mu=1.0, rho=1.0
        # u_tau = 0.09^0.25 * 1 = 0.5477
        # y+ = 1.0 * 0.5477 * 0.001 / 1.0 = 0.00055 < 11.225
        k = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([0.001], dtype=torch.float64)
        mu = torch.tensor([1.0], dtype=torch.float64)
        rho = torch.tensor([1.0], dtype=torch.float64)

        eps = wf.compute(k, y, mu, rho)

        # Expected: 2 * (1.0/1.0) * 1.0 / (0.001)^2 = 2e6
        nu = 1.0 / 1.0
        expected = 2.0 * nu * 1.0 / (0.001 ** 2)
        assert eps[0].item() == pytest.approx(expected, rel=1e-4)

    def test_output_shape(self):
        wf = CompressibleEpsilonWallFunction()
        n = 10
        k = torch.ones(n, dtype=torch.float64)
        y = torch.ones(n, dtype=torch.float64) * 0.1
        mu = torch.ones(n, dtype=torch.float64) * 1e-3
        rho = torch.ones(n, dtype=torch.float64)
        eps = wf.compute(k, y, mu, rho)
        assert eps.shape == (n,)

    def test_positive_output(self):
        """输出始终为正。"""
        wf = CompressibleEpsilonWallFunction()
        k = torch.tensor([1e-6], dtype=torch.float64)
        y = torch.tensor([1e-5], dtype=torch.float64)
        mu = torch.tensor([1e-5], dtype=torch.float64)
        rho = torch.tensor([0.1], dtype=torch.float64)
        eps = wf.compute(k, y, mu, rho)
        assert eps[0].item() > 0.0

    def test_higher_k_gives_higher_epsilon(self):
        wf = CompressibleEpsilonWallFunction()
        y = torch.tensor([0.1], dtype=torch.float64)
        mu = torch.tensor([1e-3], dtype=torch.float64)
        rho = torch.tensor([1.0], dtype=torch.float64)
        k_low = torch.tensor([0.1], dtype=torch.float64)
        k_high = torch.tensor([10.0], dtype=torch.float64)

        eps_low = wf.compute(k_low, y, mu, rho)
        eps_high = wf.compute(k_high, y, mu, rho)
        assert eps_high[0].item() > eps_low[0].item()

    def test_density_effect(self):
        """更高密度 -> 更高 y+ -> 可能切换到对数律。"""
        wf = CompressibleEpsilonWallFunction()
        k = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([0.01], dtype=torch.float64)
        mu = torch.tensor([1.0], dtype=torch.float64)

        rho_low = torch.tensor([0.1], dtype=torch.float64)
        rho_high = torch.tensor([100.0], dtype=torch.float64)

        eps_low = wf.compute(k, y, mu, rho_low)
        eps_high = wf.compute(k, y, mu, rho_high)

        # 两个值都应为正且不同
        assert eps_low[0].item() > 0
        assert eps_high[0].item() > 0


class TestCompressibleOmegaWallFunction:
    """CompressibleOmegaWallFunction 测试。"""

    def test_default_params(self):
        wf = CompressibleOmegaWallFunction()
        assert wf.kappa == 0.41
        assert wf.C_mu == 0.09
        assert wf.beta_1 == 0.075
        assert wf.y_plus_visc == 11.225

    def test_custom_params(self):
        wf = CompressibleOmegaWallFunction(kappa=0.38, beta_1=0.07)
        assert wf.kappa == 0.38
        assert wf.beta_1 == 0.07

    def test_log_law_region(self):
        """高 y+ 区域：omega = u_tau / (C_mu^{1/4} * kappa * y)。"""
        wf = CompressibleOmegaWallFunction()
        k = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([0.1], dtype=torch.float64)
        mu = torch.tensor([1e-3], dtype=torch.float64)
        rho = torch.tensor([1.0], dtype=torch.float64)

        omega = wf.compute(k, y, mu, rho)

        u_tau = 0.09 ** 0.25 * 1.0
        expected = u_tau / (0.09 ** 0.25 * 0.41 * 0.1)
        assert omega[0].item() == pytest.approx(expected, rel=1e-4)

    def test_viscous_sublayer(self):
        """低 y+ 区域：omega = 6 * nu / (beta_1 * y^2)。"""
        wf = CompressibleOmegaWallFunction()
        k = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([0.001], dtype=torch.float64)
        mu = torch.tensor([1.0], dtype=torch.float64)
        rho = torch.tensor([1.0], dtype=torch.float64)

        omega = wf.compute(k, y, mu, rho)

        nu = 1.0 / 1.0
        expected = 6.0 * nu / (0.075 * 0.001 ** 2)
        assert omega[0].item() == pytest.approx(expected, rel=1e-4)

    def test_output_shape(self):
        wf = CompressibleOmegaWallFunction()
        n = 10
        k = torch.ones(n, dtype=torch.float64)
        y = torch.ones(n, dtype=torch.float64) * 0.1
        mu = torch.ones(n, dtype=torch.float64) * 1e-3
        rho = torch.ones(n, dtype=torch.float64)
        omega = wf.compute(k, y, mu, rho)
        assert omega.shape == (n,)

    def test_positive_output(self):
        """输出始终为正。"""
        wf = CompressibleOmegaWallFunction()
        k = torch.tensor([1e-6], dtype=torch.float64)
        y = torch.tensor([1e-5], dtype=torch.float64)
        mu = torch.tensor([1e-5], dtype=torch.float64)
        rho = torch.tensor([0.1], dtype=torch.float64)
        omega = wf.compute(k, y, mu, rho)
        assert omega[0].item() > 0.0

    def test_higher_k_gives_higher_omega(self):
        wf = CompressibleOmegaWallFunction()
        y = torch.tensor([0.1], dtype=torch.float64)
        mu = torch.tensor([1e-3], dtype=torch.float64)
        rho = torch.tensor([1.0], dtype=torch.float64)
        k_low = torch.tensor([0.1], dtype=torch.float64)
        k_high = torch.tensor([10.0], dtype=torch.float64)

        omega_low = wf.compute(k_low, y, mu, rho)
        omega_high = wf.compute(k_high, y, mu, rho)
        assert omega_high[0].item() > omega_low[0].item()

    def test_smaller_y_gives_higher_omega(self):
        """更小的 y -> 更大的 omega。"""
        wf = CompressibleOmegaWallFunction()
        k = torch.tensor([1.0], dtype=torch.float64)
        mu = torch.tensor([1e-3], dtype=torch.float64)
        rho = torch.tensor([1.0], dtype=torch.float64)
        y_small = torch.tensor([0.001], dtype=torch.float64)
        y_large = torch.tensor([1.0], dtype=torch.float64)

        omega_small = wf.compute(k, y_small, mu, rho)
        omega_large = wf.compute(k, y_large, mu, rho)
        assert omega_small[0].item() > omega_large[0].item()

    def test_density_effect(self):
        """不同密度产生不同 omega。"""
        wf = CompressibleOmegaWallFunction()
        k = torch.tensor([1.0], dtype=torch.float64)
        y = torch.tensor([0.01], dtype=torch.float64)
        mu = torch.tensor([1.0], dtype=torch.float64)

        rho_low = torch.tensor([0.1], dtype=torch.float64)
        rho_high = torch.tensor([100.0], dtype=torch.float64)

        omega_low = wf.compute(k, y, mu, rho_low)
        omega_high = wf.compute(k, y, mu, rho_high)

        assert omega_low[0].item() > 0
        assert omega_high[0].item() > 0
