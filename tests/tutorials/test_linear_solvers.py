"""
Tutorial validation: linear solver smoke tests.

验证线性求解器的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestLinearSolverSmoke:
    """线性求解器 smoke 测试。"""

    def test_pcg_import(self):
        """PCG 求解器可导入。"""
        from pyfoam.solvers import PCGSolver
        assert PCGSolver is not None

    def test_pbicgstab_import(self):
        """PBiCGSTAB 求解器可导入。"""
        from pyfoam.solvers import PBiCGSTABSolver
        assert PBiCGSTABSolver is not None

    def test_gamg_import(self):
        """GAMG 求解器可导入。"""
        from pyfoam.solvers import GAMGSolver
        assert GAMGSolver is not None

    def test_smooth_solver_import(self):
        """SmoothSolver 求解器可导入。"""
        from pyfoam.solvers import SmoothSolver
        assert SmoothSolver is not None

    def test_diagonal_solver_import(self):
        """Diagonal 求解器可导入。"""
        from pyfoam.solvers import DiagonalSolver
        assert DiagonalSolver is not None


class TestPreconditionerSmoke:
    """预条件器 smoke 测试。"""

    def test_dic_import(self):
        """DIC 预条件器可导入。"""
        from pyfoam.solvers import DICPreconditioner
        assert DICPreconditioner is not None

    def test_dilu_import(self):
        """DILU 预条件器可导入。"""
        from pyfoam.solvers import DILUPreconditioner
        assert DILUPreconditioner is not None

    def test_ilu0_import(self):
        """ILU0 预条件器可导入。"""
        from pyfoam.solvers import ILU0Preconditioner
        assert ILU0Preconditioner is not None


class TestSmootherSmoke:
    """平滑器 smoke 测试。"""

    def test_gauss_seidel_import(self):
        """Gauss-Seidel 平滑器可导入。"""
        from pyfoam.solvers import GaussSeidelSmoother
        assert GaussSeidelSmoother is not None

    def test_jacobi_import(self):
        """Jacobi 平滑器可导入。"""
        from pyfoam.solvers import JacobiSmoother
        assert JacobiSmoother is not None

    def test_dicg_import(self):
        """DICG 平滑器可导入。"""
        from pyfoam.solvers import DICGSmoother
        assert DICGSmoother is not None
