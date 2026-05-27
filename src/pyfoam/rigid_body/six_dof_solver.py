"""
6DOF rigid body motion solver.

Solves the 6-degree-of-freedom equations of motion for a rigid body
under gravity and external forces/moments.  Orientation is represented
as a unit quaternion (Hamilton convention: ``q = w + xi + yj + zk``).

In OpenFOAM, the ``sixDoFRigidBodyMotion`` class handles rigid body
dynamics for FSI simulations.  This module provides a standalone Python
equivalent with:

- Symplectic Euler integration (first-order, energy-preserving)
- Runge-Kutta 4 integration (fourth-order accurate)
- Configurable gravity, mass, inertia tensor
- External force/moment accumulation

Usage::

    solver = SixDoFSolver(
        mass=1.0,
        inertia=torch.tensor([1.0, 1.0, 1.0]),
        position=torch.tensor([0.0, 0.0, 0.0]),
        gravity=torch.tensor([0.0, -9.81, 0.0]),
    )
    solver.add_force(torch.tensor([10.0, 0.0, 0.0]))
    solver.step(dt=0.001, method="symplectic_euler")
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["SixDoFSolver"]


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton quaternion product ``q1 * q2``.

    Convention: ``q = (w, x, y, z)`` where ``w`` is the scalar part.

    Args:
        q1: ``(4,)`` quaternion.
        q2: ``(4,)`` quaternion.

    Returns:
        ``(4,)`` quaternion product.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate ``(w, -x, -y, -z)``."""
    return torch.stack([q[0], -q[1], -q[2], -q[3]])


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length."""
    norm = q.norm()
    if norm < 1e-12:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=q.dtype, device=q.device)
    return q / norm


def _quat_rotate_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate 3-D vector ``v`` by unit quaternion ``q``.

    Uses the formula: ``v' = q * (0, v) * q^{-1}``.

    Args:
        q: ``(4,)`` unit quaternion ``(w, x, y, z)``.
        v: ``(3,)`` vector to rotate.

    Returns:
        ``(3,)`` rotated vector.
    """
    qv = torch.cat([torch.zeros(1, dtype=v.dtype, device=v.device), v])
    return _quat_multiply(_quat_multiply(q, qv), _quat_conjugate(q))[1:]


def _quat_from_angular_velocity(
    omega: torch.Tensor, dt: float
) -> torch.Tensor:
    """Create a quaternion increment from angular velocity.

    Uses the small-angle approximation for one time step::

        angle = |omega| * dt
        axis  = omega / |omega|   (if |omega| > eps)
        dq    = (cos(angle/2), sin(angle/2) * axis)

    Args:
        omega: ``(3,)`` angular velocity vector.
        dt: Time step.

    Returns:
        ``(4,)`` quaternion increment.
    """
    angle = omega.norm() * dt
    if angle < 1e-12:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=omega.dtype, device=omega.device)
    axis = omega / omega.norm()
    half = angle / 2.0
    return torch.cat([
        torch.cos(half).unsqueeze(0),
        torch.sin(half) * axis,
    ])


def _quat_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion to 3x3 rotation matrix.

    Args:
        q: ``(4,)`` unit quaternion ``(w, x, y, z)``.

    Returns:
        ``(3, 3)`` rotation matrix.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


# ---------------------------------------------------------------------------
# SixDoFSolver
# ---------------------------------------------------------------------------


class SixDoFSolver:
    """Six-degree-of-freedom rigid body motion solver.

    Tracks position, velocity, orientation (quaternion), and angular
    velocity of a rigid body under gravity and external forces/moments.

    Inertia is specified as principal-axis moments ``(Ixx, Iyy, Izz)``
    (diagonal inertia tensor in the body frame).

    Attributes:
        position: ``(3,)`` centre-of-mass position in world frame.
        velocity: ``(3,)`` centre-of-mass velocity in world frame.
        orientation: ``(4,)`` unit quaternion ``(w, x, y, z)``.
        angular_velocity: ``(3,)`` angular velocity in body frame.
    """

    def __init__(
        self,
        mass: float = 1.0,
        inertia: torch.Tensor | None = None,
        position: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        orientation: torch.Tensor | None = None,
        angular_velocity: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
    ) -> None:
        """Initialise the 6DOF solver.

        Args:
            mass: Body mass (kg).
            inertia: ``(3,)`` principal moments of inertia ``(Ixx, Iyy, Izz)``.
                Defaults to identity (1, 1, 1).
            position: ``(3,)`` initial position.  Defaults to origin.
            velocity: ``(3,)`` initial velocity.  Defaults to zero.
            orientation: ``(4,)`` initial quaternion ``(w, x, y, z)``.
                Defaults to identity quaternion.
            angular_velocity: ``(3,)`` initial angular velocity (body frame).
                Defaults to zero.
            gravity: ``(3,)`` gravitational acceleration.  Defaults to zero.
        """
        device = get_device()
        dtype = get_default_dtype()

        self._mass: float = mass
        self._inertia: torch.Tensor = (
            inertia.to(device=device, dtype=dtype)
            if inertia is not None
            else torch.ones(3, device=device, dtype=dtype)
        )

        self._position: torch.Tensor = (
            position.to(device=device, dtype=dtype)
            if position is not None
            else torch.zeros(3, device=device, dtype=dtype)
        )
        self._velocity: torch.Tensor = (
            velocity.to(device=device, dtype=dtype)
            if velocity is not None
            else torch.zeros(3, device=device, dtype=dtype)
        )
        self._orientation: torch.Tensor = _quat_normalize(
            orientation.to(device=device, dtype=dtype)
            if orientation is not None
            else torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        )
        self._angular_velocity: torch.Tensor = (
            angular_velocity.to(device=device, dtype=dtype)
            if angular_velocity is not None
            else torch.zeros(3, device=device, dtype=dtype)
        )
        self._gravity: torch.Tensor = (
            gravity.to(device=device, dtype=dtype)
            if gravity is not None
            else torch.zeros(3, device=device, dtype=dtype)
        )

        # Accumulated external forces and moments (reset each step)
        self._force_accumulator: torch.Tensor = torch.zeros(
            3, device=device, dtype=dtype
        )
        self._moment_accumulator: torch.Tensor = torch.zeros(
            3, device=device, dtype=dtype
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mass(self) -> float:
        """Return body mass."""
        return self._mass

    @property
    def inertia(self) -> torch.Tensor:
        """Return principal moments of inertia."""
        return self._inertia

    @property
    def position(self) -> torch.Tensor:
        """Return centre-of-mass position."""
        return self._position

    @property
    def velocity(self) -> torch.Tensor:
        """Return centre-of-mass velocity."""
        return self._velocity

    @property
    def orientation(self) -> torch.Tensor:
        """Return orientation quaternion ``(w, x, y, z)``."""
        return self._orientation

    @property
    def angular_velocity(self) -> torch.Tensor:
        """Return angular velocity (body frame)."""
        return self._angular_velocity

    @property
    def gravity(self) -> torch.Tensor:
        """Return gravity vector."""
        return self._gravity

    @gravity.setter
    def gravity(self, g: torch.Tensor) -> None:
        """Set gravity vector."""
        self._gravity = g.to(device=self._position.device, dtype=self._position.dtype)

    # ------------------------------------------------------------------
    # Force / moment accumulation
    # ------------------------------------------------------------------

    def add_force(self, force: torch.Tensor) -> None:
        """Accumulate an external force in world frame.

        Args:
            force: ``(3,)`` force vector (N).
        """
        self._force_accumulator += force.to(
            device=self._force_accumulator.device,
            dtype=self._force_accumulator.dtype,
        )

    def add_moment(self, moment: torch.Tensor) -> None:
        """Accumulate an external moment in body frame.

        Args:
            moment: ``(3,)`` moment vector (N*m).
        """
        self._moment_accumulator += moment.to(
            device=self._moment_accumulator.device,
            dtype=self._moment_accumulator.dtype,
        )

    def _reset_accumulators(self) -> None:
        """Zero out force and moment accumulators."""
        self._force_accumulator.zero_()
        self._moment_accumulator.zero_()

    # ------------------------------------------------------------------
    # Derivatives (for integrators)
    # ------------------------------------------------------------------

    def _compute_derivatives(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        orientation: torch.Tensor,
        angular_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute time derivatives of the state variables.

        Returns:
            Tuple of (d_position, d_velocity, d_orientation, d_angular_velocity).
        """
        # Translational: dv/dt = F_total / m
        gravity_force = self._gravity * self._mass
        total_force = self._force_accumulator + gravity_force
        d_velocity = total_force / self._mass

        # Rotational: dω/dt = I^{-1} * (M - ω × (I * ω))
        I_omega = self._inertia * angular_velocity
        gyroscopic = torch.linalg.cross(angular_velocity, I_omega)
        net_moment = self._moment_accumulator - gyroscopic
        d_angular_velocity = net_moment / self._inertia

        return velocity, d_velocity, angular_velocity, d_angular_velocity

    # ------------------------------------------------------------------
    # Integration methods
    # ------------------------------------------------------------------

    def _step_symplectic_euler(self, dt: float) -> None:
        """Symplectic Euler integration (first-order, semi-implicit).

        Updates velocity first, then position — preserves energy better
        than explicit Euler for Hamiltonian systems.
        """
        dp, dv, dw_quat, domega = self._compute_derivatives(
            self._position, self._velocity, self._orientation, self._angular_velocity
        )

        # Update velocity first (semi-implicit)
        self._velocity = self._velocity + dv * dt
        self._angular_velocity = self._angular_velocity + domega * dt

        # Then update position and orientation using new velocities
        self._position = self._position + self._velocity * dt

        dq = _quat_from_angular_velocity(self._angular_velocity, dt)
        self._orientation = _quat_normalize(_quat_multiply(dq, self._orientation))

    def _step_rk4(self, dt: float) -> None:
        """Runge-Kutta 4 integration (fourth-order accurate)."""
        pos = self._position
        vel = self._velocity
        ori = self._orientation
        omega = self._angular_velocity

        # k1
        dp1, dv1, _, domega1 = self._compute_derivatives(pos, vel, ori, omega)

        # k2
        pos2 = pos + 0.5 * dt * dp1
        vel2 = vel + 0.5 * dt * dv1
        dq2 = _quat_from_angular_velocity(omega + 0.5 * dt * domega1, 0.5 * dt)
        ori2 = _quat_normalize(_quat_multiply(dq2, ori))
        omega2 = omega + 0.5 * dt * domega1
        dp2, dv2, _, domega2 = self._compute_derivatives(pos2, vel2, ori2, omega2)

        # k3
        pos3 = pos + 0.5 * dt * dp2
        vel3 = vel + 0.5 * dt * dv2
        dq3 = _quat_from_angular_velocity(omega + 0.5 * dt * domega2, 0.5 * dt)
        ori3 = _quat_normalize(_quat_multiply(dq3, ori))
        omega3 = omega + 0.5 * dt * domega2
        dp3, dv3, _, domega3 = self._compute_derivatives(pos3, vel3, ori3, omega3)

        # k4
        pos4 = pos + dt * dp3
        vel4 = vel + dt * dv3
        dq4 = _quat_from_angular_velocity(omega + dt * domega3, dt)
        ori4 = _quat_normalize(_quat_multiply(dq4, ori))
        omega4 = omega + dt * domega3
        dp4, dv4, _, domega4 = self._compute_derivatives(pos4, vel4, ori4, omega4)

        # Combine
        self._position = pos + (dt / 6.0) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)
        self._velocity = vel + (dt / 6.0) * (dv1 + 2 * dv2 + 2 * dv3 + dv4)
        self._angular_velocity = omega + (dt / 6.0) * (domega1 + 2 * domega2 + 2 * domega3 + domega4)

        # Orientation: accumulate total angular displacement
        total_omega = omega + (dt / 6.0) * (domega1 + 2 * domega2 + 2 * domega3 + domega4)
        dq_total = _quat_from_angular_velocity(total_omega, dt)
        self._orientation = _quat_normalize(_quat_multiply(dq_total, ori))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, dt: float, method: str = "symplectic_euler") -> None:
        """Advance the simulation by one time step.

        Args:
            dt: Time step (s).
            method: Integration method — ``"symplectic_euler"`` (default)
                or ``"rk4"``.

        Raises:
            ValueError: If *method* is not recognised.
        """
        if method == "symplectic_euler":
            self._step_symplectic_euler(dt)
        elif method == "rk4":
            self._step_rk4(dt)
        else:
            raise ValueError(
                f"Unknown integration method '{method}'. "
                f"Use 'symplectic_euler' or 'rk4'."
            )

        # Re-normalize orientation to prevent drift
        self._orientation = _quat_normalize(self._orientation)

        # Reset force/moment accumulators for next step
        self._reset_accumulators()

    def translate(self, displacement: torch.Tensor) -> None:
        """Translate the body by a displacement vector.

        Args:
            displacement: ``(3,)`` translation vector.
        """
        self._position += displacement.to(
            device=self._position.device, dtype=self._position.dtype
        )

    def rotate(self, axis: torch.Tensor, angle: float) -> None:
        """Rotate the body about an axis by a given angle.

        Args:
            axis: ``(3,)`` rotation axis (will be normalised).
            angle: Rotation angle in radians.
        """
        axis = axis.to(device=self._position.device, dtype=self._position.dtype)
        norm = axis.norm()
        if norm < 1e-12:
            return
        axis = axis / norm
        half = torch.tensor(angle / 2.0, dtype=self._position.dtype, device=self._position.device)
        dq = torch.cat([
            torch.cos(half).unsqueeze(0),
            torch.sin(half) * axis,
        ])
        self._orientation = _quat_normalize(_quat_multiply(dq, self._orientation))

    def kinetic_energy(self) -> torch.Tensor:
        """Return total kinetic energy (translational + rotational)."""
        ke_trans = 0.5 * self._mass * self._velocity.dot(self._velocity)
        I_omega = self._inertia * self._angular_velocity
        ke_rot = 0.5 * self._angular_velocity.dot(I_omega)
        return ke_trans + ke_rot

    def rotation_matrix(self) -> torch.Tensor:
        """Return the 3x3 rotation matrix from body to world frame."""
        return _quat_to_rotation_matrix(self._orientation)

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return a snapshot of the current state."""
        return {
            "position": self._position.clone(),
            "velocity": self._velocity.clone(),
            "orientation": self._orientation.clone(),
            "angular_velocity": self._angular_velocity.clone(),
        }

    def __repr__(self) -> str:
        return (
            f"SixDoFSolver(mass={self._mass}, "
            f"pos={self._position.tolist()}, "
            f"vel={self._velocity.tolist()})"
        )
