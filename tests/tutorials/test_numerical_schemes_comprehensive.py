"""
Tutorial validation: solver numerical scheme comprehensive tests.

全面验证求解器数值格式。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestNumericalSchemeComprehensive:
    """全面数值格式测试。"""

    def test_all_interpolation_schemes(self):
        """所有插值格式可导入。"""
        from pyfoam.discretisation import (
            LinearInterpolation, UpwindInterpolation, LinearUpwindInterpolation,
            CubicInterpolation, VanLeerInterpolation, MUSCLInterpolation,
            GammaInterpolation, HarmonicInterpolation, MidPointInterpolation,
            QuickInterpolation, SFCDInterpolation, LUSTInterpolation,
            ClippedLinearInterpolation, BlendedInterpolation,
        )
        assert LinearInterpolation is not None
        assert UpwindInterpolation is not None
        assert LinearUpwindInterpolation is not None
        assert CubicInterpolation is not None
        assert VanLeerInterpolation is not None
        assert MUSCLInterpolation is not None
        assert GammaInterpolation is not None
        assert HarmonicInterpolation is not None
        assert MidPointInterpolation is not None
        assert QuickInterpolation is not None
        assert SFCDInterpolation is not None
        assert LUSTInterpolation is not None
        assert ClippedLinearInterpolation is not None
        assert BlendedInterpolation is not None

    def test_all_gradient_schemes(self):
        """所有梯度格式可导入。"""
        from pyfoam.discretisation import (
            GaussLinearGrad, LeastSquaresGrad, FourthGrad,
            CellLimitedGrad, FaceLimitedGrad,
        )
        assert GaussLinearGrad is not None
        assert LeastSquaresGrad is not None
        assert FourthGrad is not None
        assert CellLimitedGrad is not None
        assert FaceLimitedGrad is not None

    def test_all_sngrad_schemes(self):
        """所有 snGrad 格式可导入。"""
        from pyfoam.discretisation import (
            CorrectedSnGrad, UncorrectedSnGrad, BoundedSnGrad,
            LimitedSnGrad, OrthogonalSnGrad, OverRelaxedSnGrad,
        )
        assert CorrectedSnGrad is not None
        assert UncorrectedSnGrad is not None
        assert BoundedSnGrad is not None
        assert LimitedSnGrad is not None
        assert OrthogonalSnGrad is not None
        assert OverRelaxedSnGrad is not None

    def test_all_ddt_schemes(self):
        """所有时间格式可导入。"""
        from pyfoam.discretisation import (
            EulerDdt, CrankNicolsonDdt, BackwardDdt, SteadyStateDdt, BoundedDdt,
        )
        assert EulerDdt is not None
        assert CrankNicolsonDdt is not None
        assert BackwardDdt is not None
        assert SteadyStateDdt is not None
        assert BoundedDdt is not None

    def test_scheme_registry(self):
        """格式注册表可用。"""
        from pyfoam.discretisation import create_ddt_scheme, sn_grad_from_name
        assert create_ddt_scheme is not None
        assert sn_grad_from_name is not None
