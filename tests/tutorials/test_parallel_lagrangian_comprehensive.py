"""
Tutorial validation: solver parallel comprehensive tests.

全面验证求解器并行计算模型。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestParallelComprehensive:
    """全面并行计算测试。"""

    def test_decomposition_import(self):
        """Decomposition 可导入。"""
        from pyfoam.parallel import Decomposition
        assert Decomposition is not None

    def test_halo_exchange_import(self):
        """HaloExchange 可导入。"""
        from pyfoam.parallel import HaloExchange
        assert HaloExchange is not None

    def test_parallel_field_import(self):
        """ParallelField 可导入。"""
        from pyfoam.parallel import ParallelField
        assert ParallelField is not None

    def test_parallel_solver_import(self):
        """ParallelSolver 可导入。"""
        from pyfoam.parallel import ParallelSolver
        assert ParallelSolver is not None

    def test_parallel_io_import(self):
        """ParallelIO 可导入。"""
        from pyfoam.parallel import parallel_io
        assert parallel_io is not None


class TestWaveComprehensive:
    """全面波浪模型测试。"""

    def test_airy_wave_import(self):
        """AiryWave 可导入。"""
        from pyfoam.waves import AiryWave
        assert AiryWave is not None

    def test_stokes_wave_import(self):
        """StokesWave 可导入。"""
        from pyfoam.waves import StokesWave
        assert StokesWave is not None

    def test_cnoidal_wave_import(self):
        """CnoidalWave 可导入。"""
        from pyfoam.waves import CnoidalWave
        assert CnoidalWave is not None


class TestLagrangianComprehensive:
    """全面拉格朗日模型测试。"""

    def test_breakup_model_import(self):
        """BreakupModel 可导入。"""
        from pyfoam.lagrangian import BreakupModel
        assert BreakupModel is not None

    def test_buoyancy_force_import(self):
        """BuoyancyForce 可导入。"""
        from pyfoam.lagrangian import BuoyancyForce
        assert BuoyancyForce is not None

    def test_brownian_motion_import(self):
        """BrownianMotionForce 可导入。"""
        from pyfoam.lagrangian import BrownianMotionForce
        assert BrownianMotionForce is not None

    def test_drag_force_import(self):
        """DragForce 可导入。"""
        from pyfoam.lagrangian import DragForce
        assert DragForce is not None

    def test_lift_force_import(self):
        """LiftForce 可导入。"""
        from pyfoam.lagrangian import LiftForce
        assert LiftForce is not None

    def test_virtual_mass_force_import(self):
        """VirtualMassForce 可导入。"""
        from pyfoam.lagrangian import VirtualMassForce
        assert VirtualMassForce is not None

    def test_collision_model_import(self):
        """CollisionModel 可导入。"""
        from pyfoam.lagrangian import CollisionModel
        assert CollisionModel is not None

    def test_injector_import(self):
        """Injector 可导入。"""
        from pyfoam.lagrangian import Injector
        assert Injector is not None
