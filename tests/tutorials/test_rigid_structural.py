"""
Tutorial validation: rigid body and structural smoke tests.

验证刚体运动和结构力学的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestRigidBodySmoke:
    """刚体运动 smoke 测试。"""

    def test_rigid_body_solver_import(self):
        """RigidBodySolver 可导入。"""
        from pyfoam.rigid_body import RigidBodySolver
        assert RigidBodySolver is not None

    def test_six_dof_solver_import(self):
        """SixDoFSolver 可导入。"""
        from pyfoam.rigid_body import SixDoFSolver
        assert SixDoFSolver is not None

    def test_joint_import(self):
        """Joint 可导入。"""
        from pyfoam.rigid_body import Joint
        assert Joint is not None

    def test_revolute_joint_import(self):
        """RevoluteJoint 可导入。"""
        from pyfoam.rigid_body import RevoluteJoint
        assert RevoluteJoint is not None

    def test_spherical_joint_import(self):
        """SphericalJoint 可导入。"""
        from pyfoam.rigid_body import SphericalJoint
        assert SphericalJoint is not None

    def test_linear_spring_import(self):
        """LinearSpring 可导入。"""
        from pyfoam.rigid_body import LinearSpring
        assert LinearSpring is not None

    def test_linear_damper_import(self):
        """LinearDamper 可导入。"""
        from pyfoam.rigid_body import LinearDamper
        assert LinearDamper is not None


class TestStructuralSmoke:
    """结构力学 smoke 测试。"""

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
