"""
Abstract base class for motion solvers.

Provides the interface contract that all motion solver implementations
must follow.  In OpenFOAM, ``motionSolver`` is the base for all mesh
motion and rigid body dynamics solvers (e.g. ``sixDoFRigidBodyMotion``,
``velocityLaplacian``).

Usage::

    class MySolver(MotionSolver):
        def solve_displacement(self, body, dt):
            ...

    solver = MySolver()
    disp = solver.solve_displacement(body, dt=0.001)
"""

from __future__ import annotations

import abc
from typing import Any

import torch

__all__ = ["MotionSolver"]


class MotionSolver(abc.ABC):
    """Abstract base class for motion solvers.

    All motion solvers must implement :meth:`solve_displacement` which
    returns the displacement vector for a body over a time step ``dt``.
    """

    @abc.abstractmethod
    def solve_displacement(
        self, body: Any, dt: float
    ) -> torch.Tensor:
        """Compute the displacement of *body* over time step *dt*.

        Args:
            body: Object with at minimum ``position`` and ``velocity``
                attributes (each a ``(3,)`` tensor).
            dt: Time step in seconds.

        Returns:
            ``(3,)`` displacement vector (m).
        """
        ...

    def step(self, body: Any, dt: float) -> torch.Tensor:
        """Advance *body* by one time step and return displacement.

        The default implementation delegates to :meth:`solve_displacement`
        and applies the displacement to ``body.position``.

        Args:
            body: Body object (see :meth:`solve_displacement`).
            dt: Time step (s).

        Returns:
            ``(3,)`` displacement applied.
        """
        disp = self.solve_displacement(body, dt)
        body.position = body.position + disp.to(
            device=body.position.device, dtype=body.position.dtype
        )
        return disp
