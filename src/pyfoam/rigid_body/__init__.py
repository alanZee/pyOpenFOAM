"""
pyfoam.rigid_body — 6DOF rigid body dynamics solver.

Provides:

- :class:`SixDoFSolver` — six degree-of-freedom motion solver
- :class:`MotionSolver` — abstract base for motion solvers
- :class:`RigidBodySolver` — Newton-Euler rigid body solver
- :class:`Joint` hierarchy — revolute, prismatic, spherical, fixed
- :class:`Restraint` hierarchy — linear spring, linear/angular damper
"""

from pyfoam.rigid_body.six_dof_solver import SixDoFSolver
from pyfoam.rigid_body.motion_solver import MotionSolver
from pyfoam.rigid_body.solver import RigidBodySolver
from pyfoam.rigid_body.joints import (
    Joint,
    RevoluteJoint,
    PrismaticJoint,
    SphericalJoint,
    FixedJoint,
)
from pyfoam.rigid_body.restraints import (
    Restraint,
    LinearSpring,
    LinearDamper,
    AngularDamper,
)

__all__ = [
    "SixDoFSolver",
    "MotionSolver",
    "RigidBodySolver",
    "Joint",
    "RevoluteJoint",
    "PrismaticJoint",
    "SphericalJoint",
    "FixedJoint",
    "Restraint",
    "LinearSpring",
    "LinearDamper",
    "AngularDamper",
]
