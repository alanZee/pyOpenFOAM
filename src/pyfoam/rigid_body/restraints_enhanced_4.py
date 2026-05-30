"""
Enhanced restraint types v4 for rigid body motion solvers.

Extends :class:`~pyfoam.rigid_body.restraints_enhanced_3` with:

- :class:`AerodynamicRestraint` — 3D aerodynamic force with lift and drag
- :class:`ElasticFoundationRestraint` — elastic foundation (Winkler model)
- :class:`PressureRestraint` — uniform pressure force on a surface
- :class:`CentripetalRestraint` — centripetal force in rotating frames

Usage::

    aero = AerodynamicRestraint(
        air_density=1.225,
        drag_coefficient=0.3,
        lift_coefficient=0.5,
        reference_area=0.1,
        wind_velocity=torch.tensor([10.0, 0, 0]),
    )
    force = aero.force(position, velocity)

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` restraint models
"""

from __future__ import annotations

import torch

from pyfoam.rigid_body.restraints import Restraint

__all__ = [
    "AerodynamicRestraint",
    "ElasticFoundationRestraint",
    "PressureRestraint",
    "CentripetalRestraint",
]


class AerodynamicRestraint(Restraint):
    """3D aerodynamic force with separate lift and drag components.

    Models aerodynamic forces::

        F_drag = 0.5 * rho * Cd * A * |v_rel|^2 * v_hat_rel
        F_lift = 0.5 * rho * Cl * A * |v_rel|^2 * l_hat

    where l_hat is perpendicular to both v_rel and the lift axis.

    Args:
        air_density: Air density (kg/m^3).
        drag_coefficient: Drag coefficient Cd.
        lift_coefficient: Lift coefficient Cl.
        reference_area: Reference area (m^2).
        wind_velocity: ``(3,)`` wind velocity (m/s).
        lift_axis: ``(3,)`` direction perpendicular to the flow
            defining the lift direction.
    """

    def __init__(
        self,
        air_density: float = 1.225,
        drag_coefficient: float = 0.3,
        lift_coefficient: float = 0.0,
        reference_area: float = 0.1,
        wind_velocity: torch.Tensor | None = None,
        lift_axis: torch.Tensor | None = None,
    ) -> None:
        self._rho = air_density
        self._Cd = drag_coefficient
        self._Cl = lift_coefficient
        self._A = reference_area
        self._v_wind = (
            wind_velocity.to(dtype=torch.float64)
            if wind_velocity is not None
            else torch.zeros(3, dtype=torch.float64)
        )
        self._lift_axis = (
            lift_axis.to(dtype=torch.float64) / lift_axis.norm()
            if lift_axis is not None
            else torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        )

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute aerodynamic force (drag + lift).

        Args:
            position: ``(3,)`` body position (unused).
            velocity: ``(3,)`` body velocity.

        Returns:
            ``(3,)`` aerodynamic force.
        """
        vel = velocity.to(dtype=torch.float64)
        v_rel = vel - self._v_wind
        speed_rel = v_rel.norm()

        if speed_rel < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        q = 0.5 * self._rho * speed_rel ** 2  # dynamic pressure
        direction = v_rel / speed_rel

        # Drag: opposes relative motion
        drag = -q * self._Cd * self._A * direction

        # Lift: perpendicular to flow
        lift_dir = self._lift_axis - self._lift_axis.dot(direction) * direction
        lift_dir_norm = lift_dir.norm()
        if lift_dir_norm > 1e-15:
            lift_dir = lift_dir / lift_dir_norm
        else:
            lift_dir = torch.zeros(3, dtype=torch.float64)

        lift = q * self._Cl * self._A * lift_dir

        return drag + lift


class ElasticFoundationRestraint(Restraint):
    """Winkler elastic foundation restraint.

    Applies a distributed restoring force proportional to displacement::

        F = -k * A * delta

    where k is the foundation modulus, A is the contact area, and
    delta is the penetration depth.

    Args:
        foundation_modulus: Winkler foundation modulus (N/m^3).
        contact_area: Contact area (m^2).
        foundation_normal: ``(3,)`` outward normal of the foundation.
        foundation_position: Position along the normal (m).
    """

    def __init__(
        self,
        foundation_modulus: float = 1e6,
        contact_area: float = 0.01,
        foundation_normal: torch.Tensor | None = None,
        foundation_position: float = 0.0,
    ) -> None:
        self._k = foundation_modulus
        self._A = contact_area
        self._normal = (
            foundation_normal.to(dtype=torch.float64)
            / foundation_normal.norm()
            if foundation_normal is not None
            else torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        )
        self._pos = foundation_position

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute elastic foundation force.

        Args:
            position: ``(3,)`` body position.
            velocity: ``(3,)`` body velocity (unused).

        Returns:
            ``(3,)`` foundation force.
        """
        pos = position.to(dtype=torch.float64)
        proj = pos.dot(self._normal)
        penetration = self._pos - proj

        if penetration <= 0:
            return torch.zeros(3, dtype=torch.float64)

        return self._k * self._A * penetration * self._normal


class PressureRestraint(Restraint):
    """Uniform pressure force on a surface.

    Applies a constant pressure force::

        F = p * A * n

    where p is the pressure, A is the surface area, and n is the
    surface normal.

    Args:
        pressure: Applied pressure (Pa, positive = outward).
        area: Surface area (m^2).
        normal: ``(3,)`` surface normal (will be normalised).
    """

    def __init__(
        self,
        pressure: float = 1e5,
        area: float = 0.01,
        normal: torch.Tensor | None = None,
    ) -> None:
        self._p = pressure
        self._A = area
        self._normal = (
            normal.to(dtype=torch.float64) / normal.norm()
            if normal is not None
            else torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        )

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute pressure force.

        Args:
            position: ``(3,)`` body position (unused).
            velocity: ``(3,)`` body velocity (unused).

        Returns:
            ``(3,)`` pressure force.
        """
        return self._p * self._A * self._normal


class CentripetalRestraint(Restraint):
    """Centripetal force in a rotating reference frame.

    Models the apparent force experienced in a rotating frame::

        F_cent = -m * omega x (omega x r)

    where omega is the frame angular velocity and r is the position
    relative to the rotation centre.

    Args:
        frame_angular_velocity: ``(3,)`` angular velocity of the
            rotating frame (rad/s).
        mass: Mass of the body (kg).
        rotation_centre: ``(3,)`` centre of rotation.
    """

    def __init__(
        self,
        frame_angular_velocity: torch.Tensor,
        mass: float = 1.0,
        rotation_centre: torch.Tensor | None = None,
    ) -> None:
        self._omega = frame_angular_velocity.to(dtype=torch.float64)
        self._mass = mass
        self._centre = (
            rotation_centre.to(dtype=torch.float64)
            if rotation_centre is not None
            else torch.zeros(3, dtype=torch.float64)
        )

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute centripetal force.

        Args:
            position: ``(3,)`` body position.
            velocity: ``(3,)`` body velocity (unused).

        Returns:
            ``(3,)`` centripetal force.
        """
        pos = position.to(dtype=torch.float64)
        r = pos - self._centre

        # omega x r
        cross1 = torch.linalg.cross(self._omega, r)
        # omega x (omega x r)
        cross2 = torch.linalg.cross(self._omega, cross1)

        # F = -m * omega x (omega x r)
        return -self._mass * cross2
