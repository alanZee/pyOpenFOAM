"""可微分求解器测试。"""

import pytest
import torch

from pyfoam.solvers.adjoint import DifferentiableSolver, ShapeOptimizer


class TestDifferentiableSolver:
    """DifferentiableSolver 基本功能测试。"""

    def test_import(self):
        """可微分求解器可导入。"""
        from pyfoam.solvers import DifferentiableSolver, ShapeOptimizer
        assert DifferentiableSolver is not None
        assert ShapeOptimizer is not None

    def test_class_exists(self):
        """类定义存在。"""
        assert hasattr(DifferentiableSolver, "forward")
        assert hasattr(ShapeOptimizer, "optimize")
