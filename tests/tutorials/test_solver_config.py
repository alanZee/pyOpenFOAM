"""
Tutorial validation: solver preconditioner tests.

验证求解器预条件器。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


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

    def test_ilut_import(self):
        """ILUT 预条件器可导入。"""
        from pyfoam.solvers import ILUTPreconditioner
        assert ILUTPreconditioner is not None

    def test_jacobi_import(self):
        """Jacobi 预条件器可导入。"""
        from pyfoam.solvers import JacobiPreconditioner
        assert JacobiPreconditioner is not None


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


class TestSolverConfig:
    """求解器配置测试。"""

    def test_simple_config_import(self):
        """SIMPLE 配置可导入。"""
        from pyfoam.solvers import SIMPLEConfig
        assert SIMPLEConfig is not None

    def test_piso_config_import(self):
        """PISO 配置可导入。"""
        from pyfoam.solvers import PISOConfig
        assert PISOConfig is not None

    def test_pimple_config_import(self):
        """PIMPLE 配置可导入。"""
        from pyfoam.solvers import PIMPLEConfig
        assert PIMPLEConfig is not None

    def test_convergence_info_import(self):
        """ConvergenceInfo 可导入。"""
        from pyfoam.solvers import ConvergenceInfo
        assert ConvergenceInfo is not None

    def test_residual_monitor_import(self):
        """ResidualMonitor 可导入。"""
        from pyfoam.solvers import ResidualMonitor
        assert ResidualMonitor is not None
