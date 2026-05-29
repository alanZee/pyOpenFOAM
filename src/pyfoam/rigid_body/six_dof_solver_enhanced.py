"""
Enhanced 6DOF rigid body solver with improved quaternion integration.

Extends :class:`~pyfoam.rigid_body.six_dof_solver.SixDoFSolver` with:

- Velocity-Verlet integration (second-order symplectic)
- Constraint support (position/velocity constraints applied per step)
- Multi-body coupling forces
- Energy tracking over time

Usage::

    solver = EnhancedSixDoFSolver(
        mass=1.0,
        inertia=torch.tensor([1.0, 1.0, 1.0]),
        gravity=torch.tensor([0.0, -9.81, 0.0]),
    )
    solver.add_position_constraint(torch.tensor([0.0, 1.0, 0.0]))
    solver.step(dt=0.001, method="velocity_verlet")

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` class
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import Callable, Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.rigid_body.six_dof_solver import (
    SixDoFSolver,
    _quat_multiply,
    _quat_normalize,
    _quat_from_angular_velocity,
)

__all__ = [
    "EnhancedSixDoFSolver",
    "PositionConstraint",
    "VelocityConstraint",
    "ConstraintType",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint types
# ---------------------------------------------------------------------------


class ConstraintType:
    """Constraint type identifiers."""

    EQUALITY = "equality"
    INEQUALITY = "inequality"


@dataclass
class PositionConstraint:
    """Position constraint for a rigid body.

    Attributes:
        axis: ``(3,)`` constrained axis direction (normalised).
        value: Target value along the axis.
        constraint_type: ``"equality"`` or ``"inequality"``.
        stiffness: Penalty stiffness for constraint enforcement.
    """

    axis: torch.Tensor
    value: float
    constraint_type: str = ConstraintType.EQUALITY
    stiffness: float = 1e8

    def __post_init__(self) -> None:
        norm = self.axis.norm()
        if norm > 1e-12:
            self.axis = self.axis / norm

    def correction(
        self, position: torch.Tensor
    ) -> torch.Tensor:
        """Compute constraint correction force.

        For equality constraints: ``F = -k * (dot(pos, axis) - value) * axis``.
        For inequality: correction only if constraint violated.

        Args:
            position: ``(3,)`` body position.

        Returns:
            ``(3,)`` correction force.
        """
        proj = position.dot(self.axis)
        error = proj - self.value

        if self.constraint_type == ConstraintType.INEQUALITY:
            # Only enforce if body has exceeded the limit
            if error <= 0:
                return torch.zeros(3, dtype=position.dtype, device=position.device)

        return -self.stiffness * error * self.axis


@dataclass
class VelocityConstraint:
    """Velocity constraint for a rigid body.

    Attributes:
        axis: ``(3,)`` constrained axis direction (normalised).
        max_velocity: Maximum allowed velocity magnitude along the axis.
        damping: Damping coefficient applied when constraint active.
    """

    axis: torch.Tensor
    max_velocity: float = 0.0
    damping: float = 1e4

    def __post_init__(self) -> None:
        norm = self.axis.norm()
        if norm > 1e-12:
            self.axis = self.axis / norm

    def correction(
        self, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute velocity constraint damping force.

        Args:
            velocity: ``(3,)`` body velocity.

        Returns:
            ``(3,)`` damping force.
        """
        v_along = velocity.dot(self.axis)

        if abs(v_along) <= self.max_velocity:
            return torch.zeros(3, dtype=velocity.dtype, device=velocity.device)

        excess = v_along - self.max_velocity * (1.0 if v_along > 0 else -1.0)
        return -self.damping * excess * self.axis


# ---------------------------------------------------------------------------
# Enhanced solver
# ---------------------------------------------------------------------------


class EnhancedSixDoFSolver(SixDoFSolver):
    """Enhanced 6DOF solver with constraints and Velocity-Verlet integration.

    Parameters
    ----------
    mass : float
        Body mass (kg).
    inertia : torch.Tensor, optional
        ``(3,)`` principal moments of inertia.
    position, velocity, orientation, angular_velocity : torch.Tensor, optional
        Initial state.
    gravity : torch.Tensor, optional
        Gravitational acceleration.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._position_constraints: List[PositionConstraint] = []
        self._velocity_constraints: List[VelocityConstraint] = []
        self._energy_history: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Constraint management
    # ------------------------------------------------------------------

    def add_position_constraint(self, constraint: PositionConstraint) -> None:
        """Add a position constraint.

        Args:
            constraint: :class:`PositionConstraint` instance.
        """
        self._position_constraints.append(constraint)

    def add_velocity_constraint(self, constraint: VelocityConstraint) -> None:
        """Add a velocity constraint.

        Args:
            constraint: :class:`VelocityConstraint` instance.
        """
        self._velocity_constraints.append(constraint)

    def clear_constraints(self) -> None:
        """Remove all constraints."""
        self._position_constraints.clear()
        self._velocity_constraints.clear()

    @property
    def n_position_constraints(self) -> int:
        """Number of active position constraints."""
        return len(self._position_constraints)

    @property
    def n_velocity_constraints(self) -> int:
        """Number of active velocity constraints."""
        return len(self._velocity_constraints)

    # ------------------------------------------------------------------
    # Constraint enforcement
    # ------------------------------------------------------------------

    def _apply_constraints(self) -> None:
        """Apply all position and velocity constraints.

        Position constraints add corrective forces to the accumulator.
        Velocity constraints add corrective damping forces.
        """
        for pc in self._position_constraints:
            correction = pc.correction(self._position)
            self.add_force(correction)

        for vc in self._velocity_constraints:
            correction = vc.correction(self._velocity)
            self.add_force(correction)

    # ------------------------------------------------------------------
    # Velocity-Verlet integration
    # ------------------------------------------------------------------

    def _step_velocity_verlet(self, dt: float) -> None:
        """Velocity-Verlet integration (second-order, symplectic).

        Algorithm:
        1. Half-step velocity: v(t+dt/2) = v(t) + 0.5*dt*a(t)
        2. Full-step position: x(t+dt) = x(t) + dt*v(t+dt/2)
        3. Compute new acceleration a(t+dt)
        4. Half-step velocity: v(t+dt) = v(t+dt/2) + 0.5*dt*a(t+dt)
        """
        # Compute current acceleration
        _, dv, _, domega = self._compute_derivatives(
            self._position, self._velocity, self._orientation, self._angular_velocity
        )

        # Half-step velocity
        vel_half = self._velocity + 0.5 * dt * dv
        omega_half = self._angular_velocity + 0.5 * dt * domega

        # Full-step position
        self._position = self._position + vel_half * dt

        # Update orientation
        dq = _quat_from_angular_velocity(omega_half, dt)
        self._orientation = _quat_normalize(_quat_multiply(dq, self._orientation))

        # Apply constraints before computing new acceleration
        self._apply_constraints()

        # Compute new acceleration at updated position
        _, dv_new, _, domega_new = self._compute_derivatives(
            self._position, vel_half, self._orientation, omega_half
        )

        # Complete velocity step
        self._velocity = vel_half + 0.5 * dt * dv_new
        self._angular_velocity = omega_half + 0.5 * dt * domega_new

    # ------------------------------------------------------------------
    # Energy tracking
    # ------------------------------------------------------------------

    def record_energy(self) -> None:
        """Record current kinetic energy to history."""
        self._energy_history.append(self.kinetic_energy().clone())

    @property
    def energy_history(self) -> List[torch.Tensor]:
        """List of recorded kinetic energy values."""
        return self._energy_history

    def energy_drift(self) -> float:
        """Compute relative energy drift from first recorded value.

        Returns:
            Relative drift (0 = no drift).
        """
        if len(self._energy_history) < 2:
            return 0.0
        e0 = self._energy_history[0].item()
        e_last = self._energy_history[-1].item()
        if abs(e0) < 1e-30:
            return 0.0
        return abs(e_last - e0) / abs(e0)

    # ------------------------------------------------------------------
    # Override step to add velocity_verlet
    # ------------------------------------------------------------------

    def step(self, dt: float, method: str = "symplectic_euler") -> None:
        """Advance one time step with optional constraint enforcement.

        Supports all base methods plus ``"velocity_verlet"``.

        Args:
            dt: Time step (s).
            method: Integration method name.
        """
        if method == "velocity_verlet":
            self._step_velocity_verlet(dt)
        else:
            # Apply constraints before base integration
            self._apply_constraints()
            super().step(dt, method=method)
            return

        # Re-normalize orientation
        self._orientation = _quat_normalize(self._orientation)
        # Reset accumulators
        self._reset_accumulators()

    def __repr__(self) -> str:
        return (
            f"EnhancedSixDoFSolver(mass={self._mass}, "
            f"pos={self._position.tolist()}, "
            f"constraints={len(self._position_constraints)}P/"
            f"{len(self._velocity_constraints)}V)"
        )
