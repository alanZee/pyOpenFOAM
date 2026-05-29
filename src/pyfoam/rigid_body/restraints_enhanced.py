"""
Enhanced restraint types for rigid body motion solvers.

Extends :class:`~pyfoam.rigid_body.restraints.Restraint` with:

- :class:`TorsionSpring` — ``tau = -k * (theta - theta0)``
- :class:`NonlinearSpring` — ``F = -k * |x-x0|^n * direction``
- :class:`MotorRestraint` — constant torque application
- :class:`BushingRestraint` — 6-DOF spring-damper coupling

Usage::

    motor = MotorRestraint(
        axis=torch.tensor([0, 0, 1], dtype=torch.float64),
        torque_magnitude=10.0,
    )
    tau = motor.torque(angular_velocity)

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` restraint models
"""

from __future__ import annotations

import abc
from typing import Optional

import torch

from pyfoam.rigid_body.restraints import Restraint

__all__ = [
    "TorsionSpring",
    "NonlinearSpring",
    "MotorRestraint",
    "BushingRestraint",
]


class TorsionSpring:
    """Torsion spring restraint: ``tau = -k * (theta - theta0)``.

    Applies a restoring torque proportional to the angular deviation
    from a rest angle about a specified axis. In OpenFOAM, ``torsionSpring``
    restraints provide this in ``sixDoFRigidBodyMotion``.

    Note: Implements :meth:`torque` (not :meth:`force`) since it acts
    rotationally. The :meth:`force` method returns zero.

    Args:
        axis: ``(3,)`` rotation axis (will be normalised).
        stiffness: Torsional spring constant *k* (N*m/rad).
        rest_angle: Rest angle *theta0* (rad). Defaults to 0.
    """

    def __init__(
        self,
        axis: torch.Tensor,
        stiffness: float = 1.0,
        rest_angle: float = 0.0,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._k = stiffness
        self._theta0 = rest_angle
        self._accumulated_angle: float = 0.0

    def set_accumulated_angle(self, angle: float) -> None:
        """Set the accumulated rotation angle about the axis.

        Args:
            angle: Total rotation angle in radians.
        """
        self._accumulated_angle = angle

    def torque(self, angular_velocity: torch.Tensor) -> torch.Tensor:
        """Compute torsion spring torque.

        Uses the accumulated angle to compute the restoring torque.

        Args:
            angular_velocity: ``(3,)`` angular velocity (unused, for interface).

        Returns:
            ``(3,)`` torque vector.
        """
        deviation = self._accumulated_angle - self._theta0
        return -self._k * deviation * self._axis

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Torsion spring exerts no translational force."""
        return torch.zeros(3, dtype=torch.float64)


class NonlinearSpring(Restraint):
    """Nonlinear spring: ``F = -k * |x-x0|^n * direction``.

    Models a spring with a power-law force-displacement relationship.
    When ``n=1``, this reduces to a linear spring. When ``n>1``, the
    spring stiffens with displacement (hardening). When ``n<1``, it
    softens.

    Args:
        anchor: ``(3,)`` fixed anchor point.
        stiffness: Spring constant *k*.
        exponent: Power-law exponent *n* (default 1.0 = linear).
        rest_length: Natural length (m). Defaults to 0.
    """

    def __init__(
        self,
        anchor: torch.Tensor,
        stiffness: float = 1.0,
        exponent: float = 1.0,
        rest_length: float = 0.0,
    ) -> None:
        self._anchor = anchor.to(dtype=torch.float64)
        self._k = stiffness
        self._n = exponent
        self._l0 = rest_length

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """F = -k * (|x - anchor| - l0)^n * direction."""
        diff = position.to(dtype=torch.float64) - self._anchor
        dist = diff.norm()
        if dist < 1e-15:
            return torch.zeros(3, dtype=torch.float64)
        direction = diff / dist
        extension = dist - self._l0
        magnitude = self._k * (abs(extension) ** self._n)
        if extension < 0:
            magnitude = -magnitude
        return -magnitude * direction


class MotorRestraint:
    """Motor restraint: applies constant torque about an axis.

    Models a motor driving the body at a constant torque.
    The motor always applies torque in the same direction regardless
    of the current angular velocity.

    In OpenFOAM, motor restraints model active drives in
    ``sixDoFRigidBodyMotion``.

    Note: Implements :meth:`torque` for rotational effect.
    The :meth:`force` method returns zero.

    Args:
        axis: ``(3,)`` rotation axis (will be normalised).
        torque_magnitude: Torque magnitude (N*m).
    """

    def __init__(
        self,
        axis: torch.Tensor,
        torque_magnitude: float = 1.0,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._tau_mag = torque_magnitude

    @property
    def axis(self) -> torch.Tensor:
        """Motor rotation axis."""
        return self._axis.clone()

    @property
    def torque_magnitude(self) -> float:
        """Motor torque magnitude."""
        return self._tau_mag

    def torque(self, angular_velocity: torch.Tensor) -> torch.Tensor:
        """Apply constant torque along the motor axis.

        Args:
            angular_velocity: ``(3,)`` angular velocity (unused).

        Returns:
            ``(3,)`` torque vector.
        """
        return self._tau_mag * self._axis

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Motor exerts no translational force."""
        return torch.zeros(3, dtype=torch.float64)


class BushingRestraint:
    """6-DOF bushing restraint: spring-damper in all directions.

    Models a flexible connection (bushing) between two bodies or between
    a body and ground. Provides independent stiffness and damping in
    each translational and rotational DOF.

    In OpenFOAM, ``bushing`` restraints model compliant mounts in
    ``sixDoFRigidBodyMotion``.

    Args:
        anchor: ``(3,)`` anchor point (bushing centre in world frame).
        linear_stiffness: ``(3,)`` translational stiffness (N/m) per axis.
        linear_damping: ``(3,)`` translational damping (N*s/m) per axis.
        angular_stiffness: ``(3,)`` rotational stiffness (N*m/rad) per axis.
        angular_damping: ``(3,)`` rotational damping (N*m*s/rad) per axis.
    """

    def __init__(
        self,
        anchor: torch.Tensor,
        linear_stiffness: torch.Tensor | None = None,
        linear_damping: torch.Tensor | None = None,
        angular_stiffness: torch.Tensor | None = None,
        angular_damping: torch.Tensor | None = None,
    ) -> None:
        self._anchor = anchor.to(dtype=torch.float64)
        self._k_t = (
            linear_stiffness.to(dtype=torch.float64)
            if linear_stiffness is not None
            else torch.ones(3, dtype=torch.float64) * 1e4
        )
        self._c_t = (
            linear_damping.to(dtype=torch.float64)
            if linear_damping is not None
            else torch.ones(3, dtype=torch.float64) * 1e2
        )
        self._k_r = (
            angular_stiffness.to(dtype=torch.float64)
            if angular_stiffness is not None
            else torch.ones(3, dtype=torch.float64) * 1e4
        )
        self._c_r = (
            angular_damping.to(dtype=torch.float64)
            if angular_damping is not None
            else torch.ones(3, dtype=torch.float64) * 1e2
        )

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute bushing translational force.

        F_i = -k_t_i * (x_i - anchor_i) - c_t_i * v_i

        Args:
            position: ``(3,)`` body position.
            velocity: ``(3,)`` body velocity.

        Returns:
            ``(3,)`` bushing force.
        """
        pos = position.to(dtype=torch.float64)
        vel = velocity.to(dtype=torch.float64)
        displacement = pos - self._anchor
        return -self._k_t * displacement - self._c_t * vel

    def torque(self, angular_velocity: torch.Tensor) -> torch.Tensor:
        """Compute bushing rotational torque.

        tau_i = -c_r_i * omega_i

        Args:
            angular_velocity: ``(3,)`` angular velocity.

        Returns:
            ``(3,)`` bushing torque.
        """
        omega = angular_velocity.to(dtype=torch.float64)
        return -self._c_r * omega

    @property
    def anchor(self) -> torch.Tensor:
        """Bushing anchor point."""
        return self._anchor.clone()

    @property
    def linear_stiffness(self) -> torch.Tensor:
        """Translational stiffness."""
        return self._k_t.clone()
