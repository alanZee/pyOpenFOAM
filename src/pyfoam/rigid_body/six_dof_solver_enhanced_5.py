"""
Enhanced 6DOF rigid body solver v5 with adaptive substep and energy tracking.

Extends :class:`~pyfoam.rigid_body.six_dof_solver_enhanced_4.EnhancedSixDoFSolver4` with:

- Adaptive substep count (auto-adjust based on error estimates)
- Kinetic/potential energy decomposition and tracking
- Angular momentum conservation monitoring
- Velocity damping profiles (position-dependent damping)

Usage::

    solver = EnhancedSixDoFSolver5(
        mass=1.0,
        inertia=torch.tensor([1.0, 1.0, 1.0]),
        adaptive_substeps=True,
    )
    solver.step(dt=0.001, method="adaptive_substep")
    print(f"KE: {solver.kinetic_energy():.6f}")

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
from pyfoam.rigid_body.six_dof_solver_enhanced_4 import (
    EnhancedSixDoFSolver4,
    ForceHistoryEntry,
    StabilityInfo,
)

__all__ = [
    "EnhancedSixDoFSolver5",
    "EnergyTrackingState",
    "AdaptiveSubstepConfig",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EnergyTrackingState:
    """Energy tracking state for rigid body dynamics.

    Attributes:
        kinetic: Translational kinetic energy (J).
        rotational: Rotational kinetic energy (J).
        potential: Gravitational potential energy (J).
        total: Total mechanical energy (J).
        angular_momentum: ``(3,)`` angular momentum vector.
    """

    kinetic: float = 0.0
    rotational: float = 0.0
    potential: float = 0.0
    total: float = 0.0
    angular_momentum: torch.Tensor = None

    def __post_init__(self) -> None:
        if self.angular_momentum is None:
            self.angular_momentum = torch.zeros(3, dtype=torch.float64)


@dataclass
class AdaptiveSubstepConfig:
    """Configuration for adaptive substep integration.

    Attributes:
        min_substeps: Minimum number of substeps.
        max_substeps: Maximum number of substeps.
        error_tolerance: Target error tolerance for substep adaptation.
        growth_factor: Maximum substep count growth factor per step.
    """

    min_substeps: int = 1
    max_substeps: int = 32
    error_tolerance: float = 1e-4
    growth_factor: float = 2.0


# ---------------------------------------------------------------------------
# Enhanced solver v5
# ---------------------------------------------------------------------------


class EnhancedSixDoFSolver5(EnhancedSixDoFSolver4):
    """v5 enhanced 6DOF solver with adaptive substeps and energy tracking.

    Parameters
    ----------
    mass : float
        Body mass (kg).
    inertia : torch.Tensor, optional
        ``(3,)`` principal moments of inertia.
    gravity : torch.Tensor, optional
        Gravitational acceleration.
    adaptive_substeps : bool
        Enable adaptive substep count (default False).
    adaptive_config : AdaptiveSubstepConfig, optional
        Configuration for adaptive substep behaviour.
    """

    def __init__(self, **kwargs) -> None:
        adaptive = kwargs.pop("adaptive_substeps", False)
        adaptive_config = kwargs.pop("adaptive_config", None)
        super().__init__(**kwargs)
        self._adaptive_substeps = adaptive
        self._adaptive_config = adaptive_config or AdaptiveSubstepConfig()
        self._current_substeps = self._n_substeps
        self._energy_history: List[EnergyTrackingState] = []
        self._reference_height: float = 0.0

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------

    def kinetic_energy(self) -> float:
        """Compute translational kinetic energy: KE = 0.5 * m * |v|^2.

        Returns:
            Kinetic energy (J).
        """
        return 0.5 * self._mass * self._velocity.to(dtype=torch.float64).norm().item() ** 2

    def rotational_energy(self) -> float:
        """Compute rotational kinetic energy: RE = 0.5 * I * |omega|^2.

        Returns:
            Rotational kinetic energy (J).
        """
        omega = self._angular_velocity.to(dtype=torch.float64)
        inertia = self._inertia.to(dtype=torch.float64)
        return 0.5 * (inertia * omega.pow(2)).sum().item()

    def potential_energy(self, gravity: torch.Tensor | None = None) -> float:
        """Compute gravitational potential energy: PE = m * g * h.

        Args:
            gravity: Gravitational acceleration (uses body gravity if None).

        Returns:
            Potential energy (J).
        """
        g = gravity if gravity is not None else self._gravity
        if g is None:
            return 0.0
        g_vec = g.to(dtype=torch.float64)
        pos = self._position.to(dtype=torch.float64)
        height = pos.dot(-g_vec / max(g_vec.norm(), 1e-30))
        return self._mass * g_vec.norm().item() * height.item()

    def angular_momentum(self) -> torch.Tensor:
        """Compute angular momentum: L = I * omega.

        Returns:
            ``(3,)`` angular momentum vector.
        """
        inertia = self._inertia.to(dtype=torch.float64)
        omega = self._angular_velocity.to(dtype=torch.float64)
        return inertia * omega

    def track_energy(self) -> EnergyTrackingState:
        """Compute and record the current energy state.

        Returns:
            :class:`EnergyTrackingState`.
        """
        ke = self.kinetic_energy()
        re = self.rotational_energy()
        pe = self.potential_energy()
        am = self.angular_momentum()

        state = EnergyTrackingState(
            kinetic=ke,
            rotational=re,
            potential=pe,
            total=ke + re + pe,
            angular_momentum=am,
        )
        self._energy_history.append(state)
        return state

    @property
    def energy_history(self) -> List[EnergyTrackingState]:
        """List of recorded energy states."""
        return self._energy_history

    # ------------------------------------------------------------------
    # Adaptive substep
    # ------------------------------------------------------------------

    def _estimate_substep_error(self, dt: float, n_sub: int) -> float:
        """Estimate integration error for a given substep count.

        Compares a single step with a two-substep result to estimate
        the local truncation error.

        Args:
            dt: Time step.
            n_sub: Number of substeps.

        Returns:
            Estimated relative error.
        """
        # Save state
        pos_save = self._position.clone()
        vel_save = self._velocity.clone()
        orient_save = self._orientation.clone()
        omega_save = self._angular_velocity.clone()

        # Single step
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            self._step_symplectic_lie(sub_dt)
        y_single_pos = self._position.clone()
        y_single_vel = self._velocity.clone()

        # Restore
        self._position = pos_save
        self._velocity = vel_save
        self._orientation = orient_save
        self._angular_velocity = omega_save

        # Two-substep
        n_sub2 = n_sub * 2
        sub_dt2 = dt / n_sub2
        for _ in range(n_sub2):
            self._step_symplectic_lie(sub_dt2)
        y_double_pos = self._position.clone()
        y_double_vel = self._velocity.clone()

        # Restore
        self._position = pos_save
        self._velocity = vel_save
        self._orientation = orient_save
        self._angular_velocity = omega_save

        # Error estimate (Richardson extrapolation)
        pos_err = (y_single_pos - y_double_pos).norm()
        vel_err = (y_single_vel - y_double_vel).norm()
        pos_scale = max(y_double_pos.norm().item(), 1e-10)
        vel_scale = max(y_double_vel.norm().item(), 1e-10)

        return max(
            (pos_err / pos_scale).item(),
            (vel_err / vel_scale).item(),
        )

    def _adaptive_substep_integrate(self, dt: float) -> None:
        """Integrate with adaptive substep count.

        Starts with the current substep count and adapts based on
        error estimates.

        Args:
            dt: Time step.
        """
        cfg = self._adaptive_config
        n_sub = self._current_substeps

        # Try current substep count
        error = self._estimate_substep_error(dt, n_sub)

        if error > cfg.error_tolerance and n_sub < cfg.max_substeps:
            # Increase substeps
            n_sub = min(
                int(n_sub * cfg.growth_factor),
                cfg.max_substeps,
            )
        elif error < cfg.error_tolerance * 0.5 and n_sub > cfg.min_substeps:
            # Decrease substeps
            n_sub = max(
                int(n_sub / cfg.growth_factor),
                cfg.min_substeps,
            )

        self._current_substeps = n_sub

        # Actual integration
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            self._step_symplectic_lie(sub_dt)

    # ------------------------------------------------------------------
    # Velocity damping
    # ------------------------------------------------------------------

    def apply_position_dependent_damping(
        self,
        damping_coefficient: float,
        reference_position: torch.Tensor,
        damping_range: float,
    ) -> None:
        """Apply position-dependent velocity damping.

        Damping increases linearly from 0 at the reference position
        to the full coefficient at distance ``damping_range``.

        Args:
            damping_coefficient: Maximum damping coefficient.
            reference_position: ``(3,)`` reference position.
            damping_range: Distance at which full damping applies.
        """
        pos = self._position.to(dtype=torch.float64)
        ref = reference_position.to(dtype=torch.float64)
        distance = (pos - ref).norm().item()

        damping = damping_coefficient * min(distance / max(damping_range, 1e-30), 1.0)
        factor = max(0.0, 1.0 - damping)
        self._velocity = self._velocity * factor

    # ------------------------------------------------------------------
    # Override step
    # ------------------------------------------------------------------

    def step(self, dt: float, method: str = "symplectic_euler") -> None:
        """Advance one time step with v5 integration methods.

        Supports all base methods plus ``"adaptive_substep"``.

        Args:
            dt: Time step (s).
            method: Integration method name.
        """
        if method == "adaptive_substep":
            self._apply_constraints()
            self._apply_baumgarte(dt)
            self._apply_ground_contact()
            self._apply_constraint_damping()
            self._adaptive_substep_integrate(dt)
            self._time += dt
        else:
            super().step(dt, method=method)

        self._reset_accumulators()

    def __repr__(self) -> str:
        return (
            f"EnhancedSixDoFSolver5(mass={self._mass}, "
            f"substeps={self._current_substeps}, "
            f"adaptive={self._adaptive_substeps}, "
            f"t={self._time:.4f})"
        )
