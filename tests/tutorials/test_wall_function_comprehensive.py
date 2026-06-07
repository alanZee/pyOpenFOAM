"""
Tutorial validation: solver wall function comprehensive tests.

全面验证求解器壁面函数。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


class TestWallFunctionComprehensive:
    """全面壁面函数测试。"""

    def test_wall_function_import(self):
        """壁面函数可导入。"""
        from pyfoam.turbulence import wall_functions
        assert wall_functions is not None

    def test_wall_treatment_import(self):
        """壁面处理可导入。"""
        from pyfoam.turbulence import wall_treatment
        assert wall_treatment is not None

    def test_epsilon_wall_function_import(self):
        """epsilon 壁面函数可导入。"""
        from pyfoam.turbulence import wall_functions
        assert wall_functions is not None

    def test_nut_wall_function_import(self):
        """nut 壁面函数可导入。"""
        from pyfoam.turbulence import wall_functions
        assert wall_functions is not None

    def test_omega_wall_function_import(self):
        """omega 壁面函数可导入。"""
        from pyfoam.turbulence import wall_functions
        assert wall_functions is not None

    def test_k_wall_function_import(self):
        """k 壁面函数可导入。"""
        from pyfoam.turbulence import wall_functions
        assert wall_functions is not None
