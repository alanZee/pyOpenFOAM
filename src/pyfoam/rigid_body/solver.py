"""
Newton-Euler rigid body solver.

Solves translational and rotational dynamics via Newton's second law::

    F = m * a           (translational)
    tau = I * alpha      (rotational)

Supports accumulation of external forces and torques, and provides
energy / momentum diagnostics.

In OpenFOAM, the ``sixDoFRigidBodyMotionSolver`` uses the Newton-Euler
formulation internally.  This module exposes a clean Python equivalent
that can be combined with :class:`Joint` and :class:`Restraint` objects.

Usage::

    solver = RigidBodySolver(mass=2.0, inertia=torch.tensor([1, 2, 3], dtype=torch.float64))
    solver.add_force(torch.tensor([10, 0, 0], dtype=torch.float64))
    solver.add_torque(torch.tensor([0, 0, 5], dtype=torch.float64))
    acc, alpha = solver.solve(dt=0.001)
"""

from __future__ import annotations

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["RigidBodySolver"]


class RigidBodySolver:
    """Newton-Euler rigid body solver.

    Attributes:
        mass: Body mass (kg).
        inertia: ``(3,)`` principal moments of inertia.
        position: ``(3,)`` centre-of-mass position.
        velocity: ``(3,)`` centre-of-mass velocity.
        orientation: ``(4,)`` unit quaternion ``(w, x, y, z)``.
        angular_velocity: ``(3,)`` angular velocity (body frame).
    """

    def __init__(
        self,
        mass: float = 1.0,
        inertia: torch.Tensor | None = None,
        position: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        orientation: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
    ) -> None:
        device = get_device()
        dtype = get_default_dtype()

        self._mass = mass
        self._inertia = (
            inertia.to(device=device, dtype=dtype)
            if inertia is not None
            else torch.ones(3, device=device, dtype=dtype)
        )
        self._position = (
            position.to(device=device, dtype=dtype)
            if position is not None
            else torch.zeros(3, device=device, dtype=dtype)
        )
        self._velocity = (
            velocity.to(device=device, dtype=dtype)
            if velocity is not None
            else torch.zeros(3, device=device, dtype=dtype)
        )
        self._orientation = (
            orientation.to(device=device, dtype=dtype)
            if orientation is not None
            else torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        )
        self._angular_velocity = (
            angular_velocity.to(device=device, dtype=dtype)
            if angular_velocity is not None
            else torch.zeros(3, device=device, dtype=dtype)
        )

        self._force_accum: torch.Tensor = torch.zeros(3, device=device, dtype=dtype)
        self._torque_accum: torch.Tensor = torch.zeros(3, device=device, dtype=dtype)

    # -- Properties -------------------------------------------------------

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def inertia(self) -> torch.Tensor:
        return self._inertia

    @property
    def position(self) -> torch.Tensor:
        return self._position

    @position.setter
    def position(self, val: torch.Tensor) -> None:
        self._position = val.to(device=self._position.device, dtype=self._position.dtype)

    @property
    def velocity(self) -> torch.Tensor:
        return self._velocity

    @velocity.setter
    def velocity(self, val: torch.Tensor) -> None:
        self._velocity = val.to(device=self._velocity.device, dtype=self._velocity.dtype)

    @property
    def orientation(self) -> torch.Tensor:
        return self._orientation

    @property
    def angular_velocity(self) -> torch.Tensor:
        return self._angular_velocity

    # -- Force / torque accumulation --------------------------------------

    def add_force(self, force: torch.Tensor) -> None:
        """Accumulate external force (world frame, N)."""
        self._force_accum += force.to(
            device=self._force_accum.device, dtype=self._force_accum.dtype
        )

    def add_torque(self, torque: torch.Tensor) -> None:
        """Accumulate external torque (body frame, N*m)."""
        self._torque_accum += torque.to(
            device=self._torque_accum.device, dtype=self._torque_accum.dtype
        )

    def _reset(self) -> None:
        self._force_accum.zero_()
        self._torque_accum.zero_()

    # -- Solver -----------------------------------------------------------

    def solve(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute accelerations and integrate one step.

        Uses semi-implicit (symplectic) Euler::

            v_{n+1} = v_n + a * dt
            x_{n+1} = x_n + v_{n+1} * dt

        Args:
            dt: Time step (s).

        Returns:
            ``(linear_acceleration, angular_acceleration)`` — both ``(3,)``.
        """
        # Translational: a = F / m
        linear_acc = self._force_accum / self._mass
        self._velocity = self._velocity + linear_acc * dt
        self._position = self._position + self._velocity * dt

        # Rotational: alpha = tau / I  (diagonal inertia)
        angular_acc = self._torque_accum / self._inertia
        self._angular_velocity = self._angular_velocity + angular_acc * dt

        self._reset()
        return linear_acc, angular_acc

    # -- Diagnostics ------------------------------------------------------

    def linear_momentum(self) -> torch.Tensor:
        """p = m * v."""
        return self._mass * self._velocity

    def angular_momentum(self) -> torch.Tensor:
        """L = I * omega."""
        return self._inertia * self._angular_velocity

    def kinetic_energy(self) -> torch.Tensor:
        """KE = 0.5 * m * v^2 + 0.5 * omega^T * I * omega."""
        ke_t = 0.5 * self._mass * self._velocity.dot(self._velocity)
        ke_r = 0.5 * self._angular_velocity.dot(self._inertia * self._angular_velocity)
        return ke_t + ke_r

    def __repr__(self) -> str:
        return (
            f"RigidBodySolver(mass={self._mass}, "
            f"pos={self._position.tolist()}, "
            f"vel={self._velocity.tolist()})"
        )
