"""
Tutorial validation: solver linear algebra tests.

验证求解器线性代数操作。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


class TestLinearAlgebra:
    """线性代数测试。"""

    def test_ldu_matrix_import(self):
        """LduMatrix 可导入。"""
        from pyfoam.core import LduMatrix
        assert LduMatrix is not None

    def test_fv_matrix_import(self):
        """FvMatrix 可导入。"""
        from pyfoam.core import FvMatrix
        assert FvMatrix is not None

    def test_pcg_solver_import(self):
        """PCG 求解器可导入。"""
        from pyfoam.solvers import PCGSolver
        assert PCGSolver is not None

    def test_pbicgstab_solver_import(self):
        """PBiCGSTAB 求解器可导入。"""
        from pyfoam.solvers import PBiCGSTABSolver
        assert PBiCGSTABSolver is not None

    def test_gamg_solver_import(self):
        """GAMG 求解器可导入。"""
        from pyfoam.solvers import GAMGSolver
        assert GAMGSolver is not None

    def test_smooth_solver_import(self):
        """SmoothSolver 可导入。"""
        from pyfoam.solvers import SmoothSolver
        assert SmoothSolver is not None

    def test_diagonal_solver_import(self):
        """DiagonalSolver 可导入。"""
        from pyfoam.solvers import DiagonalSolver
        assert DiagonalSolver is not None


class TestPreconditioners:
    """预条件器测试。"""

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


class TestSmoothers:
    """平滑器测试。"""

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
