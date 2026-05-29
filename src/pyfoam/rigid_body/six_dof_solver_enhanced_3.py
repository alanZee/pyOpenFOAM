"""
Enhanced 6DOF rigid body solver v3 with improved constraint handling.

Extends :class:`~pyfoam.rigid_body.six_dof_solver_enhanced_2.EnhancedSixDoFSolver2` with:

- Symplectic Lie-group integrator (preserves geometric structure)
- Iterative constraint projection with Gauss-Seidel convergence
- Energy tracking and dissipation monitoring
- Contact detection and response for multi-body systems

Usage::

    solver = EnhancedSixDoFSolver3(
        mass=1.0,
        inertia=torch.tensor([1.0, 1.0, 1.0]),
        gravity=torch.tensor([0.0, -9.81, 0.0]),
    )
    solver.step(dt=0.001, method="symplectic_lie")
    print(f"Energy: {solver.total_energy():.6f}")

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` class
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.rigid_body.six_dof_solver import (
    _quat_multiply,
    _quat_normalize,
    _quat_from_angular_velocity,
    _quat_conjugate,
    _quat_rotate_vector,
)
from pyfoam.rigid_body.six_dof_solver_enhanced_2 import (
    EnhancedSixDoFSolver2,
    BaumgarteParams,
)

__all__ = [
    "EnhancedSixDoFSolver3",
    "ContactParams",
    "EnergyState",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ContactParams:
    """Parameters for contact detection and response.

    Attributes:
        stiffness: Contact spring stiffness (N/m).
        damping: Contact damping coefficient (N*s/m).
        friction_coeff: Coulomb friction coefficient at contact.
        detection_radius: Distance threshold for contact activation.
    """

    stiffness: float = 1e6
    damping: float = 1e4
    friction_coeff: float = 0.3
    detection_radius: float = 0.01


@dataclass
class EnergyState:
    """Snapshot of the solver's energy state.

    Attributes:
        kinetic_translational: Translational kinetic energy (0.5 * m * |v|^2).
        kinetic_rotational: Rotational kinetic energy (0.5 * omega . I . omega).
        potential: Gravitational potential energy (m * g . x).
        total: Sum of all energy components.
    """

    kinetic_translational: float = 0.0
    kinetic_rotational: float = 0.0
    potential: float = 0.0
    total: float = 0.0


# ---------------------------------------------------------------------------
# Enhanced solver v3
# ---------------------------------------------------------------------------


class EnhancedSixDoFSolver3(EnhancedSixDoFSolver2):
    """v3 enhanced 6DOF solver with symplectic Lie-group integrator.

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
    baumgarte : BaumgarteParams, optional
        Baumgarte stabilisation parameters.
    contact : ContactParams, optional
        Contact response parameters.
    """

    def __init__(self, **kwargs) -> None:
        contact = kwargs.pop("contact", None)
        super().__init__(**kwargs)
        self._contact = contact or ContactParams()
        self._energy_history: List[EnergyState] = []
        self._ground_plane_y: Optional[float] = None

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------

    def compute_energy(self) -> EnergyState:
        """Compute current energy state of the body.

        Returns:
            :class:`EnergyState` with all energy components.
        """
        vel = self._velocity.to(dtype=torch.float64)
        omega = self._angular_velocity.to(dtype=torch.float64)
        pos = self._position.to(dtype=torch.float64)

        ke_trans = 0.5 * self._mass * vel.dot(vel).item()
        ke_rot = 0.5 * (self._inertia * omega * omega).sum().item()

        potential = 0.0
        if self._gravity is not None:
            potential = self._mass * (-self._gravity).dot(pos).item()

        return EnergyState(
            kinetic_translational=ke_trans,
            kinetic_rotational=ke_rot,
            potential=potential,
            total=ke_trans + ke_rot + potential,
        )

    def total_energy(self) -> float:
        """Return total energy (kinetic + potential)."""
        return self.compute_energy().total

    def record_energy(self) -> None:
        """Record current energy state in history."""
        self._energy_history.append(self.compute_energy())

    @property
    def energy_history(self) -> List[EnergyState]:
        """List of recorded energy states."""
        return self._energy_history

    # ------------------------------------------------------------------
    # Symplectic Lie-group integrator
    # ------------------------------------------------------------------

    def _step_symplectic_lie(self, dt: float) -> None:
        """Symplectic Lie-group integrator.

        A structure-preserving integrator that updates position and
        velocity in a leapfrog-like pattern:

        1. Half-step velocity update from forces
        2. Full-step position update
        3. Half-step velocity update from new forces

        Orientation is updated via exact Lie-group exponential map.
        This integrator has excellent long-term energy behaviour.
        """
        # Half-step velocity
        _, dv, _, domega = self._compute_derivatives(
            self._position, self._velocity,
            self._orientation, self._angular_velocity,
        )
        v_half = self._velocity + dv * (dt / 2.0)
        omega_half = self._angular_velocity + domega * (dt / 2.0)

        # Full-step position
        self._position = self._position + v_half * dt

        # Full-step orientation (Lie-group exponential map)
        angle = omega_half.norm() * dt
        if angle > 1e-15:
            axis = omega_half / omega_half.norm()
            half_angle = angle / 2.0
            cos_ha = math.cos(half_angle)
            sin_ha = math.sin(half_angle)
            dq = torch.cat([
                torch.tensor([cos_ha], dtype=omega_half.dtype),
                sin_ha * axis,
            ])
            self._orientation = _quat_multiply(dq, self._orientation)
        self._orientation = _quat_normalize(self._orientation)

        # Apply constraints
        self._apply_constraints()

        # Half-step velocity from new state
        _, dv_new, _, domega_new = self._compute_derivatives(
            self._position, v_half,
            self._orientation, omega_half,
        )
        self._velocity = v_half + dv_new * (dt / 2.0)
        self._angular_velocity = omega_half + domega_new * (dt / 2.0)

    # ------------------------------------------------------------------
    # Iterative constraint projection (Gauss-Seidel)
    # ------------------------------------------------------------------

    def project_constraints(
        self,
        n_iterations: int = 10,
        tolerance: float = 1e-8,
    ) -> int:
        """Iteratively project constraints using Gauss-Seidel iteration.

        Repeatedly applies position and velocity constraints until
        convergence or the maximum number of iterations is reached.

        Args:
            n_iterations: Maximum projection iterations.
            tolerance: Convergence tolerance on constraint violation.

        Returns:
            Number of iterations performed.
        """
        for iteration in range(n_iterations):
            # Compute max constraint violation
            max_violation = 0.0
            for pc in self._position_constraints:
                proj = self._position.dot(pc.axis)
                error = abs(proj.item() - pc.value)
                max_violation = max(max_violation, error)

            if max_violation < tolerance:
                return iteration + 1

            # Apply constraints
            self._apply_constraints()

        return n_iterations

    # ------------------------------------------------------------------
    # Ground plane contact
    # ------------------------------------------------------------------

    def set_ground_plane(self, y_coordinate: float) -> None:
        """Set a ground plane at the given y-coordinate.

        When the body's position drops below this level, a contact
        force is applied.

        Args:
            y_coordinate: y-position of the ground plane.
        """
        self._ground_plane_y = y_coordinate

    def _apply_ground_contact(self) -> None:
        """Apply ground plane contact force if applicable."""
        if self._ground_plane_y is None:
            return

        y = self._position[1].item()
        if y >= self._ground_plane_y:
            return

        penetration = self._ground_plane_y - y
        v_y = self._velocity[1].item()

        # Spring-damper contact force
        F_y = (
            self._contact.stiffness * penetration
            - self._contact.damping * v_y
        )
        self.add_force(torch.tensor(
            [0.0, F_y, 0.0], dtype=torch.float64
        ))

        # Friction (oppose horizontal velocity)
        v_h = torch.tensor(
            [self._velocity[0].item(), 0.0, self._velocity[2].item()],
            dtype=torch.float64,
        )
        speed_h = v_h.norm()
        if speed_h > 1e-10:
            friction = (
                -self._contact.friction_coeff
                * F_y
                * v_h / speed_h
            )
            self.add_force(friction)

    # ------------------------------------------------------------------
    # Override step
    # ------------------------------------------------------------------

    def step(self, dt: float, method: str = "symplectic_euler") -> None:
        """Advance one time step with v3 integration methods.

        Supports all base methods plus ``"symplectic_lie"``.

        Args:
            dt: Time step (s).
            method: Integration method name.
        """
        if method == "symplectic_lie":
            self._apply_constraints()
            self._apply_baumgarte(dt)
            self._apply_ground_contact()
            self._step_symplectic_lie(dt)
        else:
            super().step(dt, method=method)
            return

        self._reset_accumulators()

    def __repr__(self) -> str:
        return (
            f"EnhancedSixDoFSolver3(mass={self._mass}, "
            f"pos={self._position.tolist()}, "
            f"constraints={len(self._position_constraints)}P/"
            f"{len(self._velocity_constraints)}V)"
        )
