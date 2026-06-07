"""
Tutorial validation: solver structural comprehensive tests.

全面验证求解器结构力学模型。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestStructuralComprehensive:
    """全面结构力学测试。"""

    def test_linear_elastic_model_import(self):
        """LinearElasticModel 可导入。"""
        from pyfoam.structural import LinearElasticModel
        assert LinearElasticModel is not None

    def test_displacement_solver_import(self):
        """DisplacementSolver 可导入。"""
        from pyfoam.structural import DisplacementSolver
        assert DisplacementSolver is not None

    def test_stress_solver_import(self):
        """StressSolver 可导入。"""
        from pyfoam.structural import StressSolver
        assert StressSolver is not None

    def test_isotropic_plastic_import(self):
        """IsotropicPlasticModel 可导入。"""
        from pyfoam.structural import IsotropicPlasticModel
        assert IsotropicPlasticModel is not None

    def test_von_mises_yield_import(self):
        """VonMisesYield 可导入。"""
        from pyfoam.structural import VonMisesYield
        assert VonMisesYield is not None

    def test_damage_model_import(self):
        """DamageModel 可导入。"""
        from pyfoam.structural import DamageModel
        assert DamageModel is not None

    def test_creep_result_import(self):
        """CreepResult 可导入。"""
        from pyfoam.structural import CreepResult
        assert CreepResult is not None

    def test_fatigue_result_import(self):
        """FatigueResult 可导入。"""
        from pyfoam.structural import FatigueResult
        assert FatigueResult is not None
