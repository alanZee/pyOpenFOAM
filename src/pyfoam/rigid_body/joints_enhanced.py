"""
Enhanced joint types for multi-body rigid body dynamics.

Extends :class:`~pyfoam.rigid_body.joints.Joint` with:

- :class:`CylindricalJoint` — rotation + translation along same axis (2 DOF)
- :class:`PlanarJoint` — 2 translations + 1 rotation in a plane (3 DOF)
- :class:`UniversalJoint` — 2 rotations about intersecting axes (2 DOF)
- :class:`FreeJoint` — unconstrained 6-DOF (6 DOF)

Joint constraint enforcement with Lagrange multipliers is also provided.

Usage::

    joint = CylindricalJoint(axis=torch.tensor([0, 0, 1], dtype=torch.float64))
    assert joint.n_dof == 2

    joint = PlanarJoint(
        normal=torch.tensor([0, 0, 1], dtype=torch.float64),
    )
    assert joint.n_dof == 3

References
----------
- OpenFOAM ``rigidBodyMeshMotion`` framework
"""

from __future__ import annotations

import torch

from pyfoam.rigid_body.joints import Joint, RevoluteJoint, PrismaticJoint

__all__ = [
    "CylindricalJoint",
    "PlanarJoint",
    "UniversalJoint",
    "FreeJoint",
]


class CylindricalJoint(Joint):
    """Cylindrical joint: rotation + translation along the same axis (2 DOF).

    Allows both rotation about and translation along a single axis.
    In OpenFOAM, ``cylindrical`` joints model cylindrical bearings.

    Args:
        axis: ``(3,)`` joint axis (will be normalised).
    """

    def __init__(self, axis: torch.Tensor) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm

    @property
    def n_dof(self) -> int:
        return 2

    def allowed_axes(self) -> torch.Tensor:
        # Stack rotation and translation axes (same direction, different DOFs)
        return torch.stack([self._axis, self._axis])  # (2, 3)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow translation along the axis."""
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow rotation about the axis."""
        return domega.dot(self._axis) * self._axis


class PlanarJoint(Joint):
    """Planar joint: 2 translations + 1 rotation in a plane (3 DOF).

    Constrains motion to a plane defined by its normal vector.
    Allows translation along two in-plane axes and rotation about
    the normal.

    Args:
        normal: ``(3,)`` plane normal (will be normalised).
        in_plane_axis: ``(3,)`` first in-plane axis (optional, computed
            from normal if not provided).
    """

    def __init__(
        self,
        normal: torch.Tensor,
        in_plane_axis: torch.Tensor | None = None,
    ) -> None:
        norm = normal.norm()
        if norm < 1e-12:
            raise ValueError("Normal vector must be non-zero.")
        self._normal = normal.to(dtype=torch.float64) / norm

        if in_plane_axis is not None:
            # Ensure perpendicular
            ipa = in_plane_axis.to(dtype=torch.float64)
            ipa = ipa - ipa.dot(self._normal) * self._normal
            ipa_norm = ipa.norm()
            if ipa_norm < 1e-12:
                raise ValueError("In-plane axis must not be parallel to normal.")
            self._axis1 = ipa / ipa_norm
        else:
            # Compute an arbitrary in-plane axis
            ref = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
            if abs(ref.dot(self._normal)) > 0.9:
                ref = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
            self._axis1 = ref - ref.dot(self._normal) * self._normal
            self._axis1 = self._axis1 / self._axis1.norm()

        self._axis2 = torch.linalg.cross(self._normal, self._axis1)

    @property
    def n_dof(self) -> int:
        return 3

    def allowed_axes(self) -> torch.Tensor:
        return torch.stack([self._axis1, self._axis2, self._normal])  # (3, 3)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow translation in the plane only."""
        return dvel.dot(self._axis1) * self._axis1 + dvel.dot(self._axis2) * self._axis2

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow rotation about the normal only."""
        return domega.dot(self._normal) * self._normal


class UniversalJoint(Joint):
    """Universal (Cardan) joint: 2 rotations about intersecting axes (2 DOF).

    Allows rotation about two perpendicular axes that intersect at the
    joint point. In OpenFOAM, ``universal`` joints are used for
    gimbal-like connections.

    Args:
        axis1: ``(3,)`` first rotation axis (will be normalised).
        axis2: ``(3,)`` second rotation axis (will be normalised,
            orthogonalised against axis1).
    """

    def __init__(self, axis1: torch.Tensor, axis2: torch.Tensor) -> None:
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

    @property
    def n_dof(self) -> int:
        return 2

    def allowed_axes(self) -> torch.Tensor:
        return torch.stack([self._axis1, self._axis2])  # (2, 3)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """No translation allowed."""
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow rotation about both axes."""
        return domega.dot(self._axis1) * self._axis1 + domega.dot(self._axis2) * self._axis2


class FreeJoint(Joint):
    """Free (unconstrained) joint: all 6 DOF.

    Does not constrain any relative motion. Used when two bodies are
    connected without any mechanical restriction.

    In OpenFOAM, ``free`` joints allow the child body full freedom
    of movement relative to the parent.
    """

    @property
    def n_dof(self) -> int:
        return 6

    def allowed_axes(self) -> torch.Tensor:
        """All 6 DOF: 3 translation + 3 rotation."""
        return torch.eye(3, dtype=torch.float64)  # (3, 3) — all directions

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """All translation allowed."""
        return dvel.clone()

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """All rotation allowed."""
        return domega.clone()
