"""
Tutorial validation: ODE solver smoke tests.

验证 ODE 求解器的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestODESolverSmoke:
    """ODE 求解器 smoke 测试。"""

    def test_euler_import(self):
        """Euler ODE 求解器可导入。"""
        from pyfoam.ode import EulerSolver
        assert EulerSolver is not None

    def test_rk4_import(self):
        """RK4 ODE 求解器可导入。"""
        from pyfoam.ode import RK4Solver
        assert RK4Solver is not None

    def test_rkf45_import(self):
        """RKF45 ODE 求解器可导入。"""
        from pyfoam.ode import RKF45Solver
        assert RKF45Solver is not None

    def test_rosenbrock_import(self):
        """Rosenbrock ODE 求解器可导入。"""
        from pyfoam.ode import Rosenbrock23Solver
        assert Rosenbrock23Solver is not None

    def test_sibis_import(self):
        """SIBS ODE 求解器可导入。"""
        from pyfoam.ode import SIBSSolver
        assert SIBSSolver is not None

    def test_create_ode_solver(self):
        """ODE 求解器工厂函数可用。"""
        from pyfoam.ode import create_ode_solver
        solver = create_ode_solver("Euler")
        assert solver is not None
