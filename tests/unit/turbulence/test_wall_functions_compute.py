"""Tests for wall function compute functions.

Tests cover:
- compute_nut_u_wall (nutUWallFunction)
- compute_nut_u_rough_wall (nutURoughWallFunction)
- compute_nut_u_spalding_wall (nutUSpaldingWallFunction)
"""

import pytest
import torch

from pyfoam.turbulence.wall_functions import (
    compute_nut_u_wall,
    compute_nut_u_rough_wall,
    compute_nut_u_spalding_wall,
    compute_nut_wall,
)


class TestComputeNutUWall:
    """Tests for velocity-based nut wall function."""

    def test_shape(self):
        """返回形状正确。"""
        U = torch.ones(5, 3, dtype=torch.float64) * 10.0
        y = torch.ones(5, dtype=torch.float64) * 0.01
        result = compute_nut_u_wall(U, y, nu=1e-5)
        assert result.shape == (5,)

    def test_positive(self):
        """返回正值。"""
        U = torch.ones(8, 3, dtype=torch.float64) * 5.0
        y = torch.ones(8, dtype=torch.float64) * 0.001
        result = compute_nut_u_wall(U, y, nu=1e-5)
        assert (result > 0).all()

    def test_increases_with_velocity(self):
        """速度越大，nut 越大。"""
        y = torch.ones(3, dtype=torch.float64) * 0.01
        U_low = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        U_high = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)

        nut_low = compute_nut_u_wall(U_low, y, nu=1e-5)
        nut_high = compute_nut_u_wall(U_high, y, nu=1e-5)
        assert (nut_high > nut_low).all()

    def test_increases_with_y(self):
        """y 越大，nut 越大。"""
        U = torch.ones(3, 3, dtype=torch.float64) * 5.0
        y = torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64)

        result = compute_nut_u_wall(U, y, nu=1e-5)
        assert result[1] > result[0]
        assert result[2] > result[1]

    def test_finite_output(self):
        """输出不含 NaN/Inf。"""
        U = torch.ones(10, 3, dtype=torch.float64) * 100.0
        y = torch.ones(10, dtype=torch.float64) * 0.0001
        result = compute_nut_u_wall(U, y, nu=1e-5)
        assert torch.isfinite(result).all()

    def test_small_velocity(self):
        """极小速度不产生 NaN。"""
        U = torch.ones(3, 3, dtype=torch.float64) * 1e-6
        y = torch.ones(3, dtype=torch.float64) * 0.01
        result = compute_nut_u_wall(U, y, nu=1e-5)
        assert torch.isfinite(result).all()
        assert (result >= 0).all()


class TestComputeNutURoughWall:
    """Tests for rough-wall nut wall function."""

    def test_shape(self):
        """返回形状正确。"""
        U = torch.ones(5, 3, dtype=torch.float64) * 10.0
        y = torch.ones(5, dtype=torch.float64) * 0.01
        result = compute_nut_u_rough_wall(U, y, nu=1e-5, Ks=1e-4)
        assert result.shape == (5,)

    def test_positive(self):
        """返回正值。"""
        U = torch.ones(8, 3, dtype=torch.float64) * 5.0
        y = torch.ones(8, dtype=torch.float64) * 0.001
        result = compute_nut_u_rough_wall(U, y, nu=1e-5, Ks=1e-4)
        assert (result > 0).all()

    def test_roughness_effect(self):
        """粗糙度增大 nut。"""
        U = torch.ones(5, 3, dtype=torch.float64) * 5.0
        y = torch.ones(5, dtype=torch.float64) * 0.01

        nut_smooth = compute_nut_u_rough_wall(U, y, nu=1e-5, Ks=0.0)
        nut_rough = compute_nut_u_rough_wall(U, y, nu=1e-5, Ks=1e-3)
        assert (nut_rough >= nut_smooth).all()

    def test_increases_with_velocity(self):
        """速度越大，nut 越大。"""
        y = torch.ones(3, dtype=torch.float64) * 0.01
        U_low = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        U_high = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)

        nut_low = compute_nut_u_rough_wall(U_low, y, nu=1e-5, Ks=1e-4)
        nut_high = compute_nut_u_rough_wall(U_high, y, nu=1e-5, Ks=1e-4)
        assert (nut_high > nut_low).all()

    def test_finite_output(self):
        """输出不含 NaN/Inf。"""
        U = torch.ones(10, 3, dtype=torch.float64) * 50.0
        y = torch.ones(10, dtype=torch.float64) * 0.001
        result = compute_nut_u_rough_wall(U, y, nu=1e-5, Ks=1e-4, Cs=0.5)
        assert torch.isfinite(result).all()


class TestComputeNutUSpaldingWall:
    """Tests for Spalding unified wall function."""

    def test_shape(self):
        """返回形状正确。"""
        U = torch.ones(5, 3, dtype=torch.float64) * 10.0
        y = torch.ones(5, dtype=torch.float64) * 0.01
        result = compute_nut_u_spalding_wall(U, y, nu=1e-5)
        assert result.shape == (5,)

    def test_positive(self):
        """返回正值。"""
        U = torch.ones(8, 3, dtype=torch.float64) * 5.0
        y = torch.ones(8, dtype=torch.float64) * 0.001
        result = compute_nut_u_spalding_wall(U, y, nu=1e-5)
        assert (result > 0).all()

    def test_increases_with_velocity(self):
        """速度越大，nut 越大。"""
        y = torch.ones(3, dtype=torch.float64) * 0.01
        U_low = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        U_high = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)

        nut_low = compute_nut_u_spalding_wall(U_low, y, nu=1e-5)
        nut_high = compute_nut_u_spalding_wall(U_high, y, nu=1e-5)
        assert (nut_high > nut_low).all()

    def test_log_law_consistency(self):
        """在 log-law 区域（大 y+）与标准 nutU 结果接近。"""
        # 高 Re 流动，确保在 log-law 区
        U = torch.ones(3, 3, dtype=torch.float64) * 20.0
        y = torch.ones(3, dtype=torch.float64) * 0.01

        nut_log = compute_nut_u_wall(U, y, nu=1e-5)
        nut_spald = compute_nut_u_spalding_wall(U, y, nu=1e-5)

        # 在 log-law 区两者应接近（相对误差 < 5%）
        rel_err = (nut_spald - nut_log).abs() / nut_log.clamp(min=1e-16)
        assert (rel_err < 0.05).all()

    def test_viscous_sublayer(self):
        """在粘性底层（小 y+）Spalding 仍给出有限正值。"""
        # 低 Re 流动，小 y+
        U = torch.ones(3, 3, dtype=torch.float64) * 0.1
        y = torch.ones(3, dtype=torch.float64) * 1e-5

        result = compute_nut_u_spalding_wall(U, y, nu=1e-5)
        assert torch.isfinite(result).all()
        assert (result >= 0).all()

    def test_finite_output(self):
        """输出不含 NaN/Inf。"""
        U = torch.ones(10, 3, dtype=torch.float64) * 100.0
        y = torch.ones(10, dtype=torch.float64) * 0.0001
        result = compute_nut_u_spalding_wall(U, y, nu=1e-5)
        assert torch.isfinite(result).all()
