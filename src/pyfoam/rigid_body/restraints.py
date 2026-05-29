"""
Restraint types for rigid body motion solvers.

Implements spring, damper, and angular damper restraints used in
OpenFOAM's ``sixDoFRigidBodyMotion`` to resist or constrain body
motion via force/torque feedback:

- :class:`LinearSpring` — ``F = -k * (x - x0)``
- :class:`LinearDamper` — ``F = -c * v``
- :class:`AngularDamper` — ``tau = -c * omega``

Usage::

    spring = LinearSpring(
        anchor=torch.tensor([0, 0, 0], dtype=torch.float64),
        stiffness=100.0,
        rest_length=1.0,
    )
    force = spring.force(position, velocity)
"""

from __future__ import annotations

import abc

import torch

__all__ = [
    "Restraint",
    "LinearSpring",
    "LinearDamper",
    "AngularDamper",
]


class Restraint(abc.ABC):
    """Abstract base for restraint forces/torques."""

    @abc.abstractmethod
    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute restraint force given current state.

        Args:
            position: ``(3,)`` body centre-of-mass position.
            velocity: ``(3,)`` body centre-of-mass velocity.

        Returns:
            ``(3,)`` force vector (N).
        """
        ...


class LinearSpring(Restraint):
    """Linear spring restraint: ``F = -k * (x - x0)``.

    Models a spring connecting the body to a fixed anchor point.
    When ``rest_length > 0``, the spring force acts along the
    line connecting the body to the anchor and is proportional to
    the deviation from the rest length.

    In OpenFOAM, ``linearSpring`` restraints provide exactly this
    functionality in ``sixDoFRigidBodyMotion``.

    Args:
        anchor: ``(3,)`` fixed anchor point.
        stiffness: Spring constant *k* (N/m).
        rest_length: Natural length of the spring (m).  Defaults to 0.
    """

    def __init__(
        self,
        anchor: torch.Tensor,
        stiffness: float = 1.0,
        rest_length: float = 0.0,
    ) -> None:
        self._anchor = anchor.to(dtype=torch.float64)
        self._k = stiffness
        self._l0 = rest_length

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """F = -k * (|x - anchor| - l0) * direction."""
        diff = position.to(dtype=torch.float64) - self._anchor
        dist = diff.norm()
        if dist < 1e-15:
            return torch.zeros(3, dtype=torch.float64)
        direction = diff / dist
        extension = dist - self._l0
        return -self._k * extension * direction


class LinearDamper(Restraint):
    """Linear viscous damper: ``F = -c * v``.

    Provides a velocity-proportional drag force that opposes the
    body's translational motion.  In OpenFOAM, ``linearDamper``
    restraints provide this functionality.

    Args:
        coefficient: Damping coefficient *c* (N*s/m).
    """

    def __init__(self, coefficient: float = 1.0) -> None:
        self._c = coefficient

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        return -self._c * velocity.to(dtype=torch.float64)


class AngularDamper(Restraint):
    """Angular viscous damper: ``tau = -c * omega``.

    Provides an angular-velocity-proportional torque that opposes
    the body's rotational motion.  In OpenFOAM, ``angularDamper``
    restraints damp rotation.

    Note: This class implements :meth:`torque` rather than
    :meth:`force` since it returns a torque.  The :meth:`force`
    method returns zero (no translational effect).

    Args:
        coefficient: Damping coefficient *c* (N*m*s/rad).
    """

    def __init__(self, coefficient: float = 1.0) -> None:
        self._c = coefficient

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Angular damper exerts no translational force."""
        return torch.zeros(3, dtype=torch.float64)

    def torque(self, angular_velocity: torch.Tensor) -> torch.Tensor:
        """tau = -c * omega.

        Args:
            angular_velocity: ``(3,)`` angular velocity (rad/s).

        Returns:
            ``(3,)`` damping torque (N*m).
        """
        return -self._c * angular_velocity.to(dtype=torch.float64)
