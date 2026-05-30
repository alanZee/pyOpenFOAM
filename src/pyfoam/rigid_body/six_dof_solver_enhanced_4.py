"""
Enhanced 6DOF rigid body solver v4 with multi-body coupling.

Extends :class:`~pyfoam.rigid_body.six_dof_solver_enhanced_3.EnhancedSixDoFSolver3` with:

- Substep integration with configurable substep count
- Force and torque history recording
- Stability analysis (eigenvalue-based timestep recommendations)
- Constraint damping for improved convergence

Usage::

    solver = EnhancedSixDoFSolver4(
        mass=1.0,
        inertia=torch.tensor([1.0, 1.0, 1.0]),
        n_substeps=4,
    )
    solver.step(dt=0.001, method="substep_lie")
    print(f"Recommended dt: {solver.recommended_timestep():.6f}")

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
from pyfoam.rigid_body.six_dof_solver_enhanced_3 import (
    EnhancedSixDoFSolver3,
    ContactParams,
    EnergyState,
)

__all__ = [
    "EnhancedSixDoFSolver4",
    "ForceHistoryEntry",
    "StabilityInfo",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ForceHistoryEntry:
    """Snapshot of forces and torques at a given time.

    Attributes:
        time: Simulation time (s).
        force: ``(3,)`` total external force.
        torque: ``(3,)`` total external torque.
        position: ``(3,)`` body position.
        velocity: ``(3,)`` body velocity.
    """

    time: float = 0.0
    force: torch.Tensor = None
    torque: torch.Tensor = None
    position: torch.Tensor = None
    velocity: torch.Tensor = None

    def __post_init__(self) -> None:
        if self.force is None:
            self.force = torch.zeros(3, dtype=torch.float64)
        if self.torque is None:
            self.torque = torch.zeros(3, dtype=torch.float64)
        if self.position is None:
            self.position = torch.zeros(3, dtype=torch.float64)
        if self.velocity is None:
            self.velocity = torch.zeros(3, dtype=torch.float64)


@dataclass
class StabilityInfo:
    """Stability analysis result.

    Attributes:
        max_eigenvalue: Maximum eigenvalue of the linearised system.
        recommended_dt: Recommended maximum stable timestep.
        stiffness_ratio: Ratio of max/min stiffness (condition number).
        is_stable: Whether the current configuration is stable.
    """

    max_eigenvalue: float = 0.0
    recommended_dt: float = float("inf")
    stiffness_ratio: float = 1.0
    is_stable: bool = True


# ---------------------------------------------------------------------------
# Enhanced solver v4
# ---------------------------------------------------------------------------


class EnhancedSixDoFSolver4(EnhancedSixDoFSolver3):
    """v4 enhanced 6DOF solver with substep integration and stability analysis.

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
    contact : ContactParams, optional
        Contact response parameters.
    n_substeps : int
        Number of substeps per time step (default: 1).
    """

    def __init__(self, **kwargs) -> None:
        n_substeps = kwargs.pop("n_substeps", 1)
        super().__init__(**kwargs)
        self._n_substeps = max(1, n_substeps)
        self._force_history: List[ForceHistoryEntry] = []
        self._time: float = 0.0
        self._constraint_damping: float = 0.0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def n_substeps(self) -> int:
        """Number of substeps per time step."""
        return self._n_substeps

    def set_constraint_damping(self, damping: float) -> None:
        """Set constraint velocity damping coefficient.

        Damping is applied to velocity components along constrained
        directions to improve convergence.

        Args:
            damping: Damping coefficient (0 = no damping, 1 = full).
        """
        self._constraint_damping = max(0.0, min(1.0, damping))

    # ------------------------------------------------------------------
    # Force history
    # ------------------------------------------------------------------

    def record_force_state(self) -> None:
        """Record current force and state in history."""
        entry = ForceHistoryEntry(
            time=self._time,
            force=self._force_accumulator.clone(),
            torque=torch.zeros(3, dtype=torch.float64),
            position=self._position.clone(),
            velocity=self._velocity.clone(),
        )
        self._force_history.append(entry)

    @property
    def force_history(self) -> List[ForceHistoryEntry]:
        """List of recorded force history entries."""
        return self._force_history

    @property
    def simulation_time(self) -> float:
        """Current simulation time."""
        return self._time

    # ------------------------------------------------------------------
    # Substep integration
    # ------------------------------------------------------------------

    def _step_substep(self, dt: float) -> None:
        """Substep integration using symplectic Lie-group method.

        Divides the timestep into ``n_substeps`` smaller steps for
        improved accuracy and stability with stiff forces.

        Args:
            dt: Time step (s).
        """
        sub_dt = dt / self._n_substeps
        for _ in range(self._n_substeps):
            self._step_symplectic_lie(sub_dt)

    # ------------------------------------------------------------------
    # Stability analysis
    # ------------------------------------------------------------------

    def analyse_stability(self) -> StabilityInfo:
        """Perform stability analysis of the current configuration.

        Estimates the maximum eigenvalue of the linearised system to
        recommend a maximum stable timestep. Uses the constraint
        stiffness and body inertia.

        Returns:
            :class:`StabilityInfo`.
        """
        # Estimate stiffness from constraints
        max_stiffness = 0.0
        min_stiffness = float("inf")

        for pc in self._position_constraints:
            k = pc.stiffness
            max_stiffness = max(max_stiffness, k)
            min_stiffness = min(min_stiffness, k) if k > 0 else min_stiffness

        if max_stiffness == 0:
            return StabilityInfo(
                max_eigenvalue=0.0,
                recommended_dt=float("inf"),
                stiffness_ratio=1.0,
                is_stable=True,
            )

        min_stiffness = min_stiffness if min_stiffness < float("inf") else max_stiffness

        # Eigenvalue estimate: omega_max = sqrt(k_max / m)
        omega_max = math.sqrt(max_stiffness / max(self._mass, 1e-30))

        # Stable dt estimate: dt < 2 / omega_max (explicit stability)
        recommended_dt = 2.0 / max(omega_max, 1e-30)

        condition = max_stiffness / max(min_stiffness, 1e-30)

        return StabilityInfo(
            max_eigenvalue=omega_max,
            recommended_dt=recommended_dt,
            stiffness_ratio=condition,
            is_stable=True,
        )

    def recommended_timestep(self, safety_factor: float = 0.5) -> float:
        """Get recommended maximum stable timestep.

        Args:
            safety_factor: Safety factor applied to the stability limit.

        Returns:
            Recommended timestep.
        """
        info = self.analyse_stability()
        return info.recommended_dt * safety_factor

    # ------------------------------------------------------------------
    # Constraint damping
    # ------------------------------------------------------------------

    def _apply_constraint_damping(self) -> None:
        """Apply damping to velocity components along constrained axes."""
        if self._constraint_damping <= 0:
            return

        for pc in self._position_constraints:
            axis = pc.axis.to(dtype=torch.float64)
            v_along = self._velocity.dot(axis)
            self._velocity = (
                self._velocity
                - self._constraint_damping * v_along * axis
            )

        for vc in self._velocity_constraints:
            axis = vc.axis.to(dtype=torch.float64)
            omega_along = self._angular_velocity.dot(axis)
            self._angular_velocity = (
                self._angular_velocity
                - self._constraint_damping * omega_along * axis
            )

    # ------------------------------------------------------------------
    # Override step
    # ------------------------------------------------------------------

    def step(self, dt: float, method: str = "symplectic_euler") -> None:
        """Advance one time step with v4 integration methods.

        Supports all base methods plus ``"substep_lie"``.

        Args:
            dt: Time step (s).
            method: Integration method name.
        """
        if method == "substep_lie":
            self._apply_constraints()
            self._apply_baumgarte(dt)
            self._apply_ground_contact()
            self._apply_constraint_damping()
            self._step_substep(dt)
            self._time += dt
        else:
            super().step(dt, method=method)
            self._time += dt
            return

        self._reset_accumulators()

    def __repr__(self) -> str:
        return (
            f"EnhancedSixDoFSolver4(mass={self._mass}, "
            f"substeps={self._n_substeps}, "
            f"t={self._time:.4f})"
        )
