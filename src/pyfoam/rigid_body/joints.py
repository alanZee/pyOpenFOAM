"""
Joint types for multi-body rigid body dynamics.

Implements the four fundamental joint types used in OpenFOAM's
``rigidBodyMeshMotion`` framework:

- :class:`RevoluteJoint` — rotation about a single axis (1 DOF)
- :class:`PrismaticJoint` — translation along a single axis (1 DOF)
- :class:`SphericalJoint` — 3-DOF rotation (ball-and-socket)
- :class:`FixedJoint` — rigid connection (0 DOF)

Each joint constrains the relative motion between a parent and child
body by projecting out the constrained degrees of freedom.

Usage::

    joint = RevoluteJoint(axis=torch.tensor([0, 0, 1], dtype=torch.float64))
    allowed = joint.project_velocity(parent_vel, child_vel,
                                     parent_omega, child_omega)
"""

from __future__ import annotations

import abc

import torch

__all__ = [
    "Joint",
    "RevoluteJoint",
    "PrismaticJoint",
    "SphericalJoint",
    "FixedJoint",
]


class Joint(abc.ABC):
    """Abstract base for joint constraints.

    A joint connects two bodies and restricts their relative motion
    by projecting velocities / forces into the allowed subspace.
    """

    @property
    @abc.abstractmethod
    def n_dof(self) -> int:
        """Number of unconstrained degrees of freedom."""
        ...

    @abc.abstractmethod
    def allowed_axes(self) -> torch.Tensor:
        """Return ``(n_dof, 3)`` matrix of allowed motion directions.

        For revolute/prismatic joints this is a single axis.
        For spherical joints it is three orthogonal rotation axes.
        For fixed joints it is empty.
        """
        ...

    def project_velocity(
        self,
        parent_vel: torch.Tensor,
        child_vel: torch.Tensor,
        parent_omega: torch.Tensor,
        child_omega: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project relative velocity onto allowed subspace.

        Args:
            parent_vel: ``(3,)`` parent body linear velocity.
            child_vel: ``(3,)`` child body linear velocity.
            parent_omega: ``(3,)`` parent body angular velocity.
            child_omega: ``(3,)`` child body angular velocity.

        Returns:
            ``(allowed_linear, allowed_angular)`` — each ``(3,)``,
            the components of relative velocity that this joint permits.
        """
        dvel = child_vel - parent_vel
        domega = child_omega - parent_omega
        axes = self.allowed_axes()
        if axes.numel() == 0:
            return torch.zeros_like(dvel), torch.zeros_like(domega)
        # Determine which axes are translational vs rotational
        allowed_linear = self._project_linear(dvel, axes)
        allowed_angular = self._project_angular(domega, axes)
        return allowed_linear, allowed_angular

    def _project_linear(
        self, dvel: torch.Tensor, axes: torch.Tensor
    ) -> torch.Tensor:
        """Project relative linear velocity onto translational allowed axes."""
        result = torch.zeros_like(dvel)
        for i in range(axes.shape[0]):
            result += dvel.dot(axes[i]) * axes[i]
        return result

    def _project_angular(
        self, domega: torch.Tensor, axes: torch.Tensor
    ) -> torch.Tensor:
        """Project relative angular velocity onto rotational allowed axes."""
        result = torch.zeros_like(domega)
        for i in range(axes.shape[0]):
            result += domega.dot(axes[i]) * axes[i]
        return result

    def constraint_force(
        self,
        parent_vel: torch.Tensor,
        child_vel: torch.Tensor,
        parent_omega: torch.Tensor,
        child_omega: torch.Tensor,
        stiffness: float = 1e6,
        damping: float = 1e3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute constraint force/torque to enforce the joint.

        Uses a penalty-based approach: the constrained (disallowed)
        components of relative velocity are resisted by spring-damper
        forces proportional to *stiffness* and *damping*.

        Args:
            parent_vel, child_vel, parent_omega, child_omega: Body velocities.
            stiffness: Penalty spring constant.
            damping: Penalty damping coefficient.

        Returns:
            ``(constraint_force, constraint_torque)`` applied to child body.
        """
        dvel = child_vel - parent_vel
        domega = child_omega - parent_omega
        axes = self.allowed_axes()

        if axes.numel() == 0:
            # Fixed joint: resist all relative motion
            return -stiffness * dvel - damping * dvel, -stiffness * domega - damping * domega

        # Compute disallowed components
        allowed_linear = self._project_linear(dvel, axes)
        disallowed_linear = dvel - allowed_linear
        allowed_angular = self._project_angular(domega, axes)
        disallowed_angular = domega - allowed_angular

        cf = -stiffness * disallowed_linear - damping * disallowed_linear
        ct = -stiffness * disallowed_angular - damping * disallowed_angular
        return cf, ct


class RevoluteJoint(Joint):
    """Rotation about a single axis (1 DOF).

    In OpenFOAM, ``revolute`` joints constrain all motion except
    rotation about the specified axis.

    Args:
        axis: ``(3,)`` rotation axis (will be normalised).
    """

    def __init__(self, axis: torch.Tensor) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)  # (1, 3)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Only allow rotation about the joint axis."""
        return domega.dot(self._axis) * self._axis

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """No translation allowed."""
        return torch.zeros_like(dvel)


class PrismaticJoint(Joint):
    """Translation along a single axis (1 DOF).

    In OpenFOAM, ``prismatic`` joints constrain all motion except
    linear translation along the specified axis.

    Args:
        axis: ``(3,)`` translation axis (will be normalised).
    """

    def __init__(self, axis: torch.Tensor) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)  # (1, 3)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Only allow translation along the joint axis."""
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """No rotation allowed."""
        return torch.zeros_like(domega)


class SphericalJoint(Joint):
    """3-DOF rotation (ball-and-socket).

    Constrains all translational relative motion but allows free
    rotation about any axis.  In OpenFOAM, ``spherical`` joints are
    used for ball-and-socket connections.
    """

    def __init__(self) -> None:
        self._axes = torch.eye(3, dtype=torch.float64)

    @property
    def n_dof(self) -> int:
        return 3

    def allowed_axes(self) -> torch.Tensor:
        return self._axes  # (3, 3)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """No translation allowed."""
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """All rotation allowed."""
        return domega.clone()


class FixedJoint(Joint):
    """Rigid connection (0 DOF).

    Constrains all relative motion between parent and child bodies.
    In OpenFOAM, ``fixed`` joints weld two bodies together.
    """

    @property
    def n_dof(self) -> int:
        return 0

    def allowed_axes(self) -> torch.Tensor:
        return torch.empty(0, 3, dtype=torch.float64)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)
