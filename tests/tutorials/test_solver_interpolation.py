"""
Tutorial validation: solver interpolation tests.

验证求解器插值格式。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


class TestInterpolation:
    """插值格式测试。"""

    def test_linear_import(self):
        """线性插值可导入。"""
        from pyfoam.discretisation import LinearInterpolation
        assert LinearInterpolation is not None

    def test_upwind_import(self):
        """迎风插值可导入。"""
        from pyfoam.discretisation import UpwindInterpolation
        assert UpwindInterpolation is not None

    def test_linear_upwind_import(self):
        """线性迎风插值可导入。"""
        from pyfoam.discretisation import LinearUpwindInterpolation
        assert LinearUpwindInterpolation is not None

    def test_cubic_import(self):
        """三次插值可导入。"""
        from pyfoam.discretisation import CubicInterpolation
        assert CubicInterpolation is not None

    def test_van_leer_import(self):
        """Van Leer 插值可导入。"""
        from pyfoam.discretisation import VanLeerInterpolation
        assert VanLeerInterpolation is not None

    def test_muscl_import(self):
        """MUSCL 插值可导入。"""
        from pyfoam.discretisation import MUSCLInterpolation
        assert MUSCLInterpolation is not None

    def test_gamma_import(self):
        """Gamma 插值可导入。"""
        from pyfoam.discretisation import GammaInterpolation
        assert GammaInterpolation is not None

    def test_harmonic_import(self):
        """调和插值可导入。"""
        from pyfoam.discretisation import HarmonicInterpolation
        assert HarmonicInterpolation is not None

    def test_midpoint_import(self):
        """中点插值可导入。"""
        from pyfoam.discretisation import MidPointInterpolation
        assert MidPointInterpolation is not None

    def test_quick_import(self):
        """QUICK 插值可导入。"""
        from pyfoam.discretisation import QuickInterpolation
        assert QuickInterpolation is not None

    def test_sfcd_import(self):
        """SFCD 插值可导入。"""
        from pyfoam.discretisation import SFCDInterpolation
        assert SFCDInterpolation is not None

    def test_lust_import(self):
        """LUST 插值可导入。"""
        from pyfoam.discretisation import LUSTInterpolation
        assert LUSTInterpolation is not None
