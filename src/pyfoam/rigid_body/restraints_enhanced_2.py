"""
Enhanced restraint types v2 for rigid body motion solvers.

Extends :class:`~pyfoam.rigid_body.restraints_enhanced` with:

- :class:`CoulombFriction` — velocity-dependent friction force
- :class:`HydraulicDamper` — velocity-squared damping
- :class:`StopRestraint` — hard stop at a position limit
- :class:`PIDRestraint` — PID controller-based force

Usage::

    friction = CoulombFriction(
        normal_force=100.0,
        mu_static=0.3,
        mu_kinetic=0.2,
    )
    force = friction.force(position, velocity)

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` restraint models
"""

from __future__ import annotations

import torch

from pyfoam.rigid_body.restraints import Restraint

__all__ = [
    "CoulombFriction",
    "HydraulicDamper",
    "StopRestraint",
    "PIDRestraint",
]


class CoulombFriction:
    """Coulomb friction restraint: velocity-dependent friction force.

    Models static and kinetic friction opposing the direction of motion.
    When the body is nearly stationary (|v| < v_tol), the static friction
    coefficient is used. Otherwise, the kinetic coefficient applies.

    Args:
        normal_force: Normal contact force (N).
        mu_static: Static friction coefficient.
        mu_kinetic: Kinetic friction coefficient.
        v_tol: Velocity tolerance for static/kinetic transition (m/s).
    """

    def __init__(
        self,
        normal_force: float = 1.0,
        mu_static: float = 0.3,
        mu_kinetic: float = 0.2,
        v_tol: float = 1e-4,
    ) -> None:
        self._N = normal_force
        self._mu_s = mu_static
        self._mu_k = mu_kinetic
        self._v_tol = v_tol

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute friction force opposing motion direction.

        Args:
            position: ``(3,)`` body position (unused).
            velocity: ``(3,)`` body velocity.

        Returns:
            ``(3,)`` friction force.
        """
        vel = velocity.to(dtype=torch.float64)
        speed = vel.norm()

        if speed < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        direction = vel / speed

        if speed < self._v_tol:
            # Static friction (opposes any applied force up to mu_s * N)
            return -self._mu_s * self._N * direction
        else:
            # Kinetic friction
            return -self._mu_k * self._N * direction

    def torque(
        self, angular_velocity: torch.Tensor
    ) -> torch.Tensor:
        """Friction exerts no pure torque."""
        return torch.zeros(3, dtype=torch.float64)


class HydraulicDamper(Restraint):
    """Hydraulic damper: ``F = -c * |v| * v`` (velocity-squared damping).

    Models a hydraulic damper where the damping force scales with the
    square of velocity. Common in shock absorbers and fluid dampers.

    Args:
        coefficient: Damping coefficient *c* (N*s^2/m^2).
        min_velocity: Minimum velocity magnitude before damping activates.
    """

    def __init__(
        self,
        coefficient: float = 1.0,
        min_velocity: float = 0.0,
    ) -> None:
        self._c = coefficient
        self._v_min = min_velocity

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """F = -c * |v| * v (direction-opposing quadratic drag).

        Args:
            position: ``(3,)`` body position (unused).
            velocity: ``(3,)`` body velocity.

        Returns:
            ``(3,)`` damping force.
        """
        vel = velocity.to(dtype=torch.float64)
        speed = vel.norm()

        if speed < self._v_min:
            return torch.zeros(3, dtype=torch.float64)

        return -self._c * speed * vel


class StopRestraint(Restraint):
    """Hard stop restraint at a position limit.

    Models a physical stop that prevents the body from moving beyond
    a specified position along an axis. Uses a stiff spring-damper
    to enforce the limit.

    Args:
        axis: ``(3,)`` stop direction (will be normalised).
        limit: Position limit value along the axis.
        stiffness: Spring stiffness when limit is violated (N/m).
        damping: Damping coefficient at the stop (N*s/m).
        is_upper: If True, enforces upper limit; otherwise lower.
    """

    def __init__(
        self,
        axis: torch.Tensor,
        limit: float,
        stiffness: float = 1e6,
        damping: float = 1e4,
        is_upper: bool = True,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Stop axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._limit = limit
        self._k = stiffness
        self._c = damping
        self._is_upper = is_upper

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute stop force when position exceeds the limit.

        Args:
            position: ``(3,)`` body position.
            velocity: ``(3,)`` body velocity.

        Returns:
            ``(3,)`` stop force (zero when within limits).
        """
        pos = position.to(dtype=torch.float64)
        vel = velocity.to(dtype=torch.float64)
        proj = pos.dot(self._axis)
        v_along = vel.dot(self._axis)

        if self._is_upper:
            violation = proj - self._limit
            if violation <= 0:
                return torch.zeros(3, dtype=torch.float64)
        else:
            violation = self._limit - proj
            if violation <= 0:
                return torch.zeros(3, dtype=torch.float64)

        # Spring-damper force to push back
        sign = 1.0 if self._is_upper else -1.0
        spring_force = -sign * self._k * violation
        damper_force = -self._c * v_along
        total = spring_force + damper_force

        return total * self._axis


class PIDRestraint(Restraint):
    """PID controller-based restraint.

    Applies a force computed from a PID controller tracking a target
    position: ``F = Kp * e + Ki * integral(e) + Kd * de/dt``.

    Args:
        axis: ``(3,)`` control axis (will be normalised).
        target: Target position value along the axis.
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
    """

    def __init__(
        self,
        axis: torch.Tensor,
        target: float = 0.0,
        kp: float = 100.0,
        ki: float = 10.0,
        kd: float = 1.0,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("PID axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._target = target
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._first_call: bool = True

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute PID control force.

        Args:
            position: ``(3,)`` body position.
            velocity: ``(3,)`` body velocity.

        Returns:
            ``(3,)`` PID control force.
        """
        pos = position.to(dtype=torch.float64)
        vel = velocity.to(dtype=torch.float64)

        proj = pos.dot(self._axis).item()
        error = self._target - proj

        # Integral accumulation
        self._integral += error

        # Derivative
        if self._first_call:
            derivative = 0.0
            self._first_call = False
        else:
            derivative = error - self._prev_error

        self._prev_error = error

        # PID output
        output = (
            self._kp * error
            + self._ki * self._integral
            + self._kd * derivative
        )

        return output * self._axis

    def reset(self) -> None:
        """Reset integral and derivative state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_call = True
