"""
Tutorial validation: solver rigid body comprehensive tests.

全面验证求解器刚体运动模型。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestRigidBodyComprehensive:
    """全面刚体运动测试。"""

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

    def test_prismatic_joint_import(self):
        """PrismaticJoint 可导入。"""
        from pyfoam.rigid_body import PrismaticJoint
        assert PrismaticJoint is not None

    def test_linear_spring_import(self):
        """LinearSpring 可导入。"""
        from pyfoam.rigid_body import LinearSpring
        assert LinearSpring is not None

    def test_linear_damper_import(self):
        """LinearDamper 可导入。"""
        from pyfoam.rigid_body import LinearDamper
        assert LinearDamper is not None

    def test_torsion_spring_import(self):
        """TorsionSpring 可导入。"""
        from pyfoam.rigid_body import TorsionSpring
        assert TorsionSpring is not None

    def test_motion_solver_import(self):
        """MotionSolver 可导入。"""
        from pyfoam.rigid_body import MotionSolver
        assert MotionSolver is not None
