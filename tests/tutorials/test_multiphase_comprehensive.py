"""
Tutorial validation: solver multiphase comprehensive tests.

全面验证求解器多相流模型。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestMultiphaseComprehensive:
    """全面多相流测试。"""

    def test_inter_foam_import(self):
        """InterFoam 可导入。"""
        from pyfoam.applications import InterFoam
        assert InterFoam is not None

    def test_compressible_inter_foam_import(self):
        """CompressibleInterFoam 可导入。"""
        from pyfoam.applications import CompressibleInterFoam
        assert CompressibleInterFoam is not None

    def test_multiphase_euler_foam_import(self):
        """MultiphaseEulerFoam 可导入。"""
        from pyfoam.applications import MultiphaseEulerFoam
        assert MultiphaseEulerFoam is not None

    def test_compressible_vof_foam_import(self):
        """CompressibleVoFFoam 可导入。"""
        from pyfoam.applications import CompressibleVoFFoam
        assert CompressibleVoFFoam is not None

    def test_incompressible_vof_foam_import(self):
        """IncompressibleVoFFoam 可导入。"""
        from pyfoam.applications import IncompressibleVoFFoam
        assert IncompressibleVoFFoam is not None

    def test_cavitating_foam_import(self):
        """CavitatingFoam 可导入。"""
        from pyfoam.applications import CavitatingFoam
        assert CavitatingFoam is not None

    def test_two_phase_euler_foam_import(self):
        """TwoPhaseEulerFoam 可导入。"""
        from pyfoam.applications import TwoPhaseEulerFoam
        assert TwoPhaseEulerFoam is not None

    def test_vof_model_import(self):
        """VOF 模型可导入。"""
        from pyfoam.multiphase import CompressibleMultiphaseVoF
        assert CompressibleMultiphaseVoF is not None

    def test_surface_tension_import(self):
        """表面张力模型可导入。"""
        from pyfoam.multiphase import CSFSurfaceTension
        assert CSFSurfaceTension is not None

    def test_bubble_model_import(self):
        """Bubble 模型可导入。"""
        from pyfoam.multiphase import BubbleModel
        assert BubbleModel is not None

    def test_drag_model_import(self):
        """Drag 模型可导入。"""
        from pyfoam.multiphase import DragModel
        assert DragModel is not None

    def test_lift_model_import(self):
        """Lift 模型可导入。"""
        from pyfoam.multiphase import LiftModel
        assert LiftModel is not None
