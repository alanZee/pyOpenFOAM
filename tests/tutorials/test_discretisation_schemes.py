"""
Tutorial validation: discretisation scheme smoke tests.

验证离散格式的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestInterpolationSchemesSmoke:
    """插值格式 smoke 测试。"""

    def test_linear_import(self):
        """线性插值可导入。"""
        from pyfoam.discretisation import LinearInterpolation
        assert LinearInterpolation is not None

    def test_upwind_import(self):
        """迎风格式可导入。"""
        from pyfoam.discretisation import UpwindInterpolation
        assert UpwindInterpolation is not None

    def test_linear_upwind_import(self):
        """线性迎风格式可导入。"""
        from pyfoam.discretisation import LinearUpwindInterpolation
        assert LinearUpwindInterpolation is not None

    def test_cubic_import(self):
        """三次格式可导入。"""
        from pyfoam.discretisation import CubicInterpolation
        assert CubicInterpolation is not None

    def test_van_leer_import(self):
        """Van Leer 格式可导入。"""
        from pyfoam.discretisation import VanLeerInterpolation
        assert VanLeerInterpolation is not None

    def test_muscl_import(self):
        """MUSCL 格式可导入。"""
        from pyfoam.discretisation import MUSCLInterpolation
        assert MUSCLInterpolation is not None

    def test_gamma_import(self):
        """Gamma 格式可导入。"""
        from pyfoam.discretisation import GammaInterpolation
        assert GammaInterpolation is not None


class TestGradientSchemesSmoke:
    """梯度格式 smoke 测试。"""

    def test_gauss_linear_import(self):
        """Gauss 线性梯度可导入。"""
        from pyfoam.discretisation import GaussLinearGrad
        assert GaussLinearGrad is not None

    def test_least_squares_import(self):
        """最小二乘梯度可导入。"""
        from pyfoam.discretisation import LeastSquaresGrad
        assert LeastSquaresGrad is not None

    def test_fourth_import(self):
        """四阶梯度可导入。"""
        from pyfoam.discretisation import FourthGrad
        assert FourthGrad is not None


class TestLaplacianSchemesSmoke:
    """拉普拉斯格式 smoke 测试。"""

    def test_corrected_import(self):
        """修正 snGrad 可导入。"""
        from pyfoam.discretisation import CorrectedSnGrad
        assert CorrectedSnGrad is not None

    def test_uncorrected_import(self):
        """未修正 snGrad 可导入。"""
        from pyfoam.discretisation import UncorrectedSnGrad
        assert UncorrectedSnGrad is not None

    def test_bounded_import(self):
        """有界 snGrad 可导入。"""
        from pyfoam.discretisation import BoundedSnGrad
        assert BoundedSnGrad is not None


class TestDdtSchemesSmoke:
    """时间导数格式 smoke 测试。"""

    def test_euler_import(self):
        """Euler 时间格式可导入。"""
        from pyfoam.discretisation import EulerDdt
        assert EulerDdt is not None

    def test_crank_nicolson_import(self):
        """Crank-Nicolson 时间格式可导入。"""
        from pyfoam.discretisation import CrankNicolsonDdt
        assert CrankNicolsonDdt is not None

    def test_backward_import(self):
        """向后时间格式可导入。"""
        from pyfoam.discretisation import BackwardDdt
        assert BackwardDdt is not None

    def test_steady_state_import(self):
        """稳态时间格式可导入。"""
        from pyfoam.discretisation import SteadyStateDdt
        assert SteadyStateDdt is not None
