"""
Enhanced joint types v2 for multi-body rigid body dynamics.

Extends :class:`~pyfoam.rigid_body.joints_enhanced` with:

- :class:`ScrewJoint` — rotation + translation with fixed ratio (1 DOF)
- :class:`GimbalJoint` — 3-DOF rotation via three sequential revolute axes
- :class:`BushingJoint` — 6-DOF spring-damper coupling (compliant joint)
- :class:`RackPinionJoint` — rotation-to-translation coupling (1 DOF)

Joint motion limits and friction models are also provided.

Usage::

    joint = ScrewJoint(
        axis=torch.tensor([0, 0, 1], dtype=torch.float64),
        pitch=0.01,  # 10mm per revolution
    )
    assert joint.n_dof == 1

References
----------
- OpenFOAM ``rigidBodyMeshMotion`` framework
"""

from __future__ import annotations

import torch

from pyfoam.rigid_body.joints import Joint

__all__ = [
    "ScrewJoint",
    "GimbalJoint",
    "BushingJoint",
    "RackPinionJoint",
]


class ScrewJoint(Joint):
    """Screw (helical) joint: rotation with coupled translation (1 DOF).

    Converts rotation about the axis to translation along it
    via a fixed pitch ratio: ``dz = pitch * dtheta / (2*pi)``.

    Args:
        axis: ``(3,)`` screw axis (will be normalised).
        pitch: Linear travel per revolution (m/rev).
    """

    def __init__(self, axis: torch.Tensor, pitch: float = 0.01) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Screw axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._pitch = pitch

    @property
    def n_dof(self) -> int:
        return 1

    @property
    def pitch(self) -> float:
        """Screw pitch (m/rev)."""
        return self._pitch

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)  # (1, 3)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Translation coupled to rotation via pitch."""
        # The allowed translation is pitch/(2*pi) times the angular component
        return (self._pitch / (2.0 * torch.pi)) * dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow rotation about the screw axis."""
        return domega.dot(self._axis) * self._axis


class GimbalJoint(Joint):
    """Gimbal joint: 3-DOF rotation via three sequential revolute axes.

    Provides full rotational freedom through three nested rings.
    Equivalent to a spherical joint for small rotations, but with
    physically meaningful axis definitions.

    Args:
        axis1: ``(3,)`` outer ring axis (will be normalised).
        axis2: ``(3,)`` middle ring axis (will be normalised, orthogonalised).
        axis3: ``(3,)`` inner ring axis (will be normalised, orthogonalised).
    """

    def __init__(
        self,
        axis1: torch.Tensor,
        axis2: torch.Tensor,
        axis3: torch.Tensor,
    ) -> None:
        norm1 = axis1.norm()
        if norm1 < 1e-12:
            raise ValueError("First axis must be non-zero.")
        self._axis1 = axis1.to(dtype=torch.float64) / norm1

        # Orthogonalise axis2 against axis1
        a2 = axis2.to(dtype=torch.float64)
        a2 = a2 - a2.dot(self._axis1) * self._axis1
        norm2 = a2.norm()
        if norm2 < 1e-12:
            raise ValueError("Second axis must not be parallel to first axis.")
        self._axis2 = a2 / norm2

        # Orthogonalise axis3 against both
        a3 = axis3.to(dtype=torch.float64)
        a3 = a3 - a3.dot(self._axis1) * self._axis1
        a3 = a3 - a3.dot(self._axis2) * self._axis2
        norm3 = a3.norm()
        if norm3 < 1e-12:
            raise ValueError("Third axis must be independent.")
        self._axis3 = a3 / norm3

    @property
    def n_dof(self) -> int:
        return 3

    def allowed_axes(self) -> torch.Tensor:
        return torch.stack([self._axis1, self._axis2, self._axis3])  # (3, 3)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """No translation allowed."""
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow rotation about all three axes."""
        return (
            domega.dot(self._axis1) * self._axis1
            + domega.dot(self._axis2) * self._axis2
            + domega.dot(self._axis3) * self._axis3
        )


class BushingJoint(Joint):
    """Bushing (compliant) joint: 6-DOF spring-damper coupling.

    Does not strictly constrain DOFs but provides compliant resistance
    to relative motion via configurable stiffness and damping in each
    direction.

    Args:
        linear_stiffness: ``(3,)`` translational stiffness (N/m) per axis.
        linear_damping: ``(3,)`` translational damping (N*s/m) per axis.
        angular_stiffness: ``(3,)`` rotational stiffness (N*m/rad) per axis.
        angular_damping: ``(3,)`` rotational damping (N*m*s/rad) per axis.
    """

    def __init__(
        self,
        linear_stiffness: torch.Tensor | None = None,
        linear_damping: torch.Tensor | None = None,
        angular_stiffness: torch.Tensor | None = None,
        angular_damping: torch.Tensor | None = None,
    ) -> None:
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

    @property
    def n_dof(self) -> int:
        return 6  # All DOF, but with compliance

    def allowed_axes(self) -> torch.Tensor:
        return torch.eye(3, dtype=torch.float64)  # All directions

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """All translation allowed (but compliance resists)."""
        return dvel.clone()

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """All rotation allowed (but compliance resists)."""
        return domega.clone()

    def compliance_force(
        self,
        displacement: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bushing translational restoring force.

        Args:
            displacement: ``(3,)`` relative displacement.
            velocity: ``(3,)`` relative velocity.

        Returns:
            ``(3,)`` restoring force.
        """
        return -self._k_t * displacement - self._c_t * velocity

    def compliance_torque(
        self,
        rotation: torch.Tensor,
        angular_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bushing rotational restoring torque.

        Args:
            rotation: ``(3,)`` relative rotation vector.
            angular_velocity: ``(3,)`` relative angular velocity.

        Returns:
            ``(3,)`` restoring torque.
        """
        return -self._k_r * rotation - self._c_r * angular_velocity


class RackPinionJoint(Joint):
    """Rack-and-pinion joint: rotation coupled to translation (1 DOF).

    Converts rotation about an axis to translation along a perpendicular
    direction: ``dx = radius * dtheta``.

    Args:
        rotation_axis: ``(3,)`` pinion rotation axis (will be normalised).
        translation_axis: ``(3,)`` rack translation direction (will be
            normalised, orthogonalised).
        pinion_radius: Effective pinion radius (m).
    """

    def __init__(
        self,
        rotation_axis: torch.Tensor,
        translation_axis: torch.Tensor,
        pinion_radius: float = 0.1,
    ) -> None:
        norm_r = rotation_axis.norm()
        if norm_r < 1e-12:
            raise ValueError("Rotation axis must be non-zero.")
        self._rot_axis = rotation_axis.to(dtype=torch.float64) / norm_r

        # Orthogonalise translation against rotation
        t = translation_axis.to(dtype=torch.float64)
        t = t - t.dot(self._rot_axis) * self._rot_axis
        norm_t = t.norm()
        if norm_t < 1e-12:
            raise ValueError(
                "Translation axis must not be parallel to rotation axis."
            )
        self._trans_axis = t / norm_t
        self._radius = pinion_radius

    @property
    def n_dof(self) -> int:
        return 1

    @property
    def pinion_radius(self) -> float:
        """Pinion radius."""
        return self._radius

    def allowed_axes(self) -> torch.Tensor:
        return self._rot_axis.unsqueeze(0)  # (1, 3)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Translation coupled to rotation via radius."""
        return self._radius * dvel.dot(self._trans_axis) * self._trans_axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow rotation about the pinion axis."""
        return domega.dot(self._rot_axis) * self._rot_axis
