"""
Tutorial validation: solver snGrad tests.

验证求解器 snGrad 格式。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestSnGrad:
    """snGrad 格式测试。"""

    def test_corrected_sngrad_import(self):
        """修正 snGrad 可导入。"""
        from pyfoam.discretisation import CorrectedSnGrad
        assert CorrectedSnGrad is not None

    def test_uncorrected_sngrad_import(self):
        """未修正 snGrad 可导入。"""
        from pyfoam.discretisation import UncorrectedSnGrad
        assert UncorrectedSnGrad is not None

    def test_bounded_sngrad_import(self):
        """有界 snGrad 可导入。"""
        from pyfoam.discretisation import BoundedSnGrad
        assert BoundedSnGrad is not None

    def test_limited_sngrad_import(self):
        """限制 snGrad 可导入。"""
        from pyfoam.discretisation import LimitedSnGrad
        assert LimitedSnGrad is not None

    def test_orthogonal_sngrad_import(self):
        """正交 snGrad 可导入。"""
        from pyfoam.discretisation import OrthogonalSnGrad
        assert OrthogonalSnGrad is not None

    def test_over_relaxed_sngrad_import(self):
        """超松弛 snGrad 可导入。"""
        from pyfoam.discretisation import OverRelaxedSnGrad
        assert OverRelaxedSnGrad is not None


class TestDdtSchemes:
    """时间离散格式测试。"""

    def test_euler_ddt_import(self):
        """Euler 时间格式可导入。"""
        from pyfoam.discretisation import EulerDdt
        assert EulerDdt is not None

    def test_crank_nicolson_ddt_import(self):
        """Crank-Nicolson 时间格式可导入。"""
        from pyfoam.discretisation import CrankNicolsonDdt
        assert CrankNicolsonDdt is not None

    def test_backward_ddt_import(self):
        """向后时间格式可导入。"""
        from pyfoam.discretisation import BackwardDdt
        assert BackwardDdt is not None

    def test_steady_state_ddt_import(self):
        """稳态时间格式可导入。"""
        from pyfoam.discretisation import SteadyStateDdt
        assert SteadyStateDdt is not None

    def test_bounded_ddt_import(self):
        """有界时间格式可导入。"""
        from pyfoam.discretisation import BoundedDdt
        assert BoundedDdt is not None
