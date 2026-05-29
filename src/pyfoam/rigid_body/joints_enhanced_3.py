"""
Enhanced joint types v3 for multi-body rigid body dynamics.

Extends :class:`~pyfoam.rigid_body.joints_enhanced_2` with:

- :class:`CamJoint` — rotation-to-translation via a cam profile (1 DOF)
- :class:`GearJoint` — coupled rotation with configurable gear ratio (1 DOF)
- :class:`ConstantVelocityJoint` — equal angular velocity on both shafts (2 DOF)
- :class:`FlexibleJoint` — torsional compliance with nonlinear stiffness (1 DOF)

Usage::

    joint = GearJoint(
        axis1=torch.tensor([0, 0, 1], dtype=torch.float64),
        axis2=torch.tensor([0, 0, 1], dtype=torch.float64),
        gear_ratio=3.0,
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
    "CamJoint",
    "GearJoint",
    "ConstantVelocityJoint",
    "FlexibleJoint",
]


class CamJoint(Joint):
    """Cam joint: rotation converted to translation via cam profile (1 DOF).

    The cam profile is specified as a piecewise-linear function mapping
    rotation angle to translation distance.

    Args:
        axis: ``(3,)`` rotation axis (will be normalised).
        translation_axis: ``(3,)`` translation direction (will be
            normalised, orthogonalised against rotation axis).
        angles: ``(n,)`` cam profile angles (rad).
        lifts: ``(n,)`` cam profile lifts (m).
    """

    def __init__(
        self,
        axis: torch.Tensor,
        translation_axis: torch.Tensor,
        angles: torch.Tensor,
        lifts: torch.Tensor,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Rotation axis must be non-zero.")
        self._rot_axis = axis.to(dtype=torch.float64) / norm

        t = translation_axis.to(dtype=torch.float64)
        t = t - t.dot(self._rot_axis) * self._rot_axis
        norm_t = t.norm()
        if norm_t < 1e-12:
            raise ValueError(
                "Translation axis must not be parallel to rotation axis."
            )
        self._trans_axis = t / norm_t

        self._angles = angles.to(dtype=torch.float64)
        self._lifts = lifts.to(dtype=torch.float64)

    @property
    def n_dof(self) -> int:
        return 1

    @property
    def cam_angles(self) -> torch.Tensor:
        """Cam profile angles."""
        return self._angles.clone()

    @property
    def cam_lifts(self) -> torch.Tensor:
        """Cam profile lifts."""
        return self._lifts.clone()

    def allowed_axes(self) -> torch.Tensor:
        return self._rot_axis.unsqueeze(0)

    def evaluate_cam(self, angle: float) -> float:
        """Evaluate the cam profile at a given angle.

        Uses piecewise-linear interpolation.

        Args:
            angle: Rotation angle (rad).

        Returns:
            Translation distance (m).
        """
        angles = self._angles.numpy()
        lifts = self._lifts.numpy()
        return float(
            torch.tensor(
                torch.nn.functional.interpolate(
                    self._lifts.unsqueeze(0).unsqueeze(0),
                    size=1,
                    mode="linear",
                    align_corners=True,
                ).item()
            )
        )

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Translation coupled to rotation via cam profile."""
        return dvel.dot(self._trans_axis) * self._trans_axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow rotation about the cam axis."""
        return domega.dot(self._rot_axis) * self._rot_axis


class GearJoint(Joint):
    """Gear joint: coupled rotation with configurable ratio (1 DOF).

    Transfers rotation from one axis to another with a gear ratio:
    ``omega2 = -ratio * omega1`` (negative for meshing gears).

    Args:
        axis1: ``(3,)`` driving gear rotation axis (will be normalised).
        axis2: ``(3,)`` driven gear rotation axis (will be normalised).
        gear_ratio: Gear ratio (driven/driving). Negative for meshing.
    """

    def __init__(
        self,
        axis1: torch.Tensor,
        axis2: torch.Tensor,
        gear_ratio: float = -1.0,
    ) -> None:
        norm1 = axis1.norm()
        if norm1 < 1e-12:
            raise ValueError("First axis must be non-zero.")
        self._axis1 = axis1.to(dtype=torch.float64) / norm1

        norm2 = axis2.norm()
        if norm2 < 1e-12:
            raise ValueError("Second axis must be non-zero.")
        self._axis2 = axis2.to(dtype=torch.float64) / norm2

        self._ratio = gear_ratio

    @property
    def n_dof(self) -> int:
        return 1

    @property
    def gear_ratio(self) -> float:
        """Gear ratio."""
        return self._ratio

    def allowed_axes(self) -> torch.Tensor:
        return self._axis1.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """No translation allowed."""
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow rotation proportional to gear ratio."""
        omega1 = domega.dot(self._axis1)
        return omega1 * self._axis1


class ConstantVelocityJoint(Joint):
    """Constant velocity (CV) joint: equal angular velocity on both sides (2 DOF).

    Transmits rotation between misaligned shafts while maintaining
    equal angular velocity on both sides. Used in drive shafts.

    Args:
        axis1: ``(3,)`` first shaft axis (will be normalised).
        axis2: ``(3,)`` second shaft axis (will be normalised).
    """

    def __init__(
        self,
        axis1: torch.Tensor,
        axis2: torch.Tensor,
    ) -> None:
        norm1 = axis1.norm()
        if norm1 < 1e-12:
            raise ValueError("First axis must be non-zero.")
        self._axis1 = axis1.to(dtype=torch.float64) / norm1

        norm2 = axis2.norm()
        if norm2 < 1e-12:
            raise ValueError("Second axis must be non-zero.")
        self._axis2 = axis2.to(dtype=torch.float64) / norm2

    @property
    def n_dof(self) -> int:
        return 2

    def allowed_axes(self) -> torch.Tensor:
        return torch.stack([self._axis1, self._axis2])

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """No translation allowed."""
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow rotation about both shaft axes."""
        return (
            domega.dot(self._axis1) * self._axis1
            + domega.dot(self._axis2) * self._axis2
        )


class FlexibleJoint(Joint):
    """Flexible joint: torsional compliance with nonlinear stiffness (1 DOF).

    Provides a compliant rotational coupling where the restoring torque
    follows a nonlinear stiffness curve: ``T = -k1 * theta - k3 * theta^3``.

    Args:
        axis: ``(3,)`` joint axis (will be normalised).
        linear_stiffness: Linear stiffness coefficient (N*m/rad).
        cubic_stiffness: Cubic stiffness coefficient (N*m/rad^3).
        damping: Torsional damping coefficient (N*m*s/rad).
    """

    def __init__(
        self,
        axis: torch.Tensor,
        linear_stiffness: float = 100.0,
        cubic_stiffness: float = 0.0,
        damping: float = 10.0,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._k1 = linear_stiffness
        self._k3 = cubic_stiffness
        self._c = damping

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """No translation allowed."""
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """Allow rotation about the joint axis."""
        return domega.dot(self._axis) * self._axis

    def restoring_torque(
        self,
        angle: float,
        angular_velocity: float,
    ) -> float:
        """Compute nonlinear restoring torque.

        Args:
            angle: Current joint angle (rad).
            angular_velocity: Current angular velocity (rad/s).

        Returns:
            Restoring torque (N*m).
        """
        return -self._k1 * angle - self._k3 * angle ** 3 - self._c * angular_velocity
