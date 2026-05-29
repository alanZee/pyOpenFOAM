"""
Enhanced restraint types v3 for rigid body motion solvers.

Extends :class:`~pyfoam.rigid_body.restraints_enhanced_2` with:

- :class:`MagneticRestraint` — magnetic attraction/repulsion force
- :class:`BouyancyRestraint` — buoyancy force for submerged bodies
- :class:`ImpactRestraint` — Hertzian contact force for impacts
- :class:`WindRestraint` — aerodynamic drag with configurable Cd

Usage::

    buoyancy = BouyancyRestraint(
        fluid_density=1000.0,
        displaced_volume=0.1,
    )
    force = buoyancy.force(position, velocity)

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` restraint models
"""

from __future__ import annotations

import torch

from pyfoam.rigid_body.restraints import Restraint

__all__ = [
    "MagneticRestraint",
    "BouyancyRestraint",
    "ImpactRestraint",
    "WindRestraint",
]


class MagneticRestraint(Restraint):
    """Magnetic dipole-dipole interaction restraint.

    Models the force between two magnetic dipoles as::

        F = (3 * mu0 / 4*pi) * (m1 . r_hat)(m2 . r_hat) / r^4

    Simplified to a spring-like model with configurable sign
    (attractive or repulsive).

    Args:
        axis: ``(3,)`` direction of the magnetic field.
        strength: Magnetic interaction strength (N*m^2).
        rest_distance: Equilibrium distance (m).
        attractive: If True, force pulls toward the magnet.
    """

    def __init__(
        self,
        axis: torch.Tensor,
        strength: float = 1.0,
        rest_distance: float = 0.1,
        attractive: bool = True,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Magnetic axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._strength = strength
        self._d0 = rest_distance
        self._sign = -1.0 if attractive else 1.0

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute magnetic interaction force.

        Args:
            position: ``(3,)`` body position.
            velocity: ``(3,)`` body velocity (unused).

        Returns:
            ``(3,)`` magnetic force.
        """
        pos = position.to(dtype=torch.float64)
        proj = pos.dot(self._axis)
        distance = abs(proj)

        if distance < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        # Force magnitude: strength / distance^2 (simplified dipole)
        force_mag = self._strength / (distance ** 2)

        # Direction: attractive pulls toward origin, repulsive pushes away
        # proj > 0 means above origin along axis; attractive should pull down
        if self._sign < 0:
            # Attractive: force toward origin (opposite to position sign)
            direction = -torch.sign(proj) * self._axis
        else:
            # Repulsive: force away from origin (same as position sign)
            direction = torch.sign(proj) * self._axis

        return force_mag * direction


class BouyancyRestraint(Restraint):
    """Buoyancy force restraint for submerged bodies.

    Applies Archimedes' principle::

        F_buoy = rho_f * V * g

    where rho_f is the fluid density, V is the displaced volume, and
    g is the gravitational acceleration.

    Args:
        fluid_density: Fluid density (kg/m^3).
        displaced_volume: Volume of fluid displaced (m^3).
        gravity: ``(3,)`` gravitational acceleration (m/s^2).
    """

    def __init__(
        self,
        fluid_density: float = 1000.0,
        displaced_volume: float = 0.1,
        gravity: torch.Tensor | None = None,
    ) -> None:
        self._rho_f = fluid_density
        self._V = displaced_volume
        self._g = (
            gravity.to(dtype=torch.float64)
            if gravity is not None
            else torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64)
        )

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute buoyancy force.

        Args:
            position: ``(3,)`` body position (unused).
            velocity: ``(3,)`` body velocity (unused).

        Returns:
            ``(3,)`` buoyancy force (opposing gravity).
        """
        # F_buoy = -rho_f * V * g (opposite to gravity)
        return -self._rho_f * self._V * self._g


class ImpactRestraint(Restraint):
    """Hertzian contact force for impact modelling.

    Uses the Hertzian contact law::

        F = k * delta^(3/2)

    where delta is the penetration depth and k is the contact stiffness.

    Args:
        contact_axis: ``(3,)`` contact normal direction.
        surface_position: Position of the surface along the axis.
        contact_stiffness: Hertzian contact stiffness (N/m^(3/2)).
        restitution_coeff: Coefficient of restitution (0 = perfectly
            inelastic, 1 = perfectly elastic).
    """

    def __init__(
        self,
        contact_axis: torch.Tensor,
        surface_position: float = 0.0,
        contact_stiffness: float = 1e8,
        restitution_coeff: float = 0.8,
    ) -> None:
        norm = contact_axis.norm()
        if norm < 1e-12:
            raise ValueError("Contact axis must be non-zero.")
        self._axis = contact_axis.to(dtype=torch.float64) / norm
        self._surface_pos = surface_position
        self._k = contact_stiffness
        self._e = restitution_coeff

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute Hertzian contact force.

        Args:
            position: ``(3,)`` body position.
            velocity: ``(3,)`` body velocity.

        Returns:
            ``(3,)`` contact force (zero when not in contact).
        """
        pos = position.to(dtype=torch.float64)
        proj = pos.dot(self._axis)

        penetration = self._surface_pos - proj
        if penetration <= 0:
            return torch.zeros(3, dtype=torch.float64)

        # Hertzian force: k * delta^(3/2)
        force_mag = self._k * (penetration ** 1.5)

        # Damping component based on restitution
        vel = velocity.to(dtype=torch.float64)
        v_normal = vel.dot(self._axis)
        if v_normal < 0:
            # Approaching: add damping
            gamma = -2.0 * torch.tensor(
                torch.log(torch.tensor(self._e)).item()
            ).item() if self._e > 1e-10 else 20.0
            damping_coeff = gamma * 0.1 * self._k
            force_mag -= damping_coeff * v_normal

        return force_mag * self._axis


class WindRestraint(Restraint):
    """Aerodynamic drag restraint with configurable Cd.

    Models wind/air resistance::

        F_drag = 0.5 * rho * Cd * A * |v_rel|^2 * v_hat_rel

    where v_rel is the velocity relative to the wind.

    Args:
        air_density: Air density (kg/m^3).
        drag_coefficient: Drag coefficient Cd.
        reference_area: Reference cross-sectional area (m^2).
        wind_velocity: ``(3,)`` wind velocity (m/s).
    """

    def __init__(
        self,
        air_density: float = 1.225,
        drag_coefficient: float = 0.5,
        reference_area: float = 0.01,
        wind_velocity: torch.Tensor | None = None,
    ) -> None:
        self._rho = air_density
        self._Cd = drag_coefficient
        self._A = reference_area
        self._v_wind = (
            wind_velocity.to(dtype=torch.float64)
            if wind_velocity is not None
            else torch.zeros(3, dtype=torch.float64)
        )

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute aerodynamic drag force.

        Args:
            position: ``(3,)`` body position (unused).
            velocity: ``(3,)`` body velocity.

        Returns:
            ``(3,)`` drag force.
        """
        vel = velocity.to(dtype=torch.float64)
        v_rel = vel - self._v_wind
        speed_rel = v_rel.norm()

        if speed_rel < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        direction = v_rel / speed_rel
        force_mag = 0.5 * self._rho * self._Cd * self._A * speed_rel ** 2

        # Drag opposes relative motion
        return -force_mag * direction
