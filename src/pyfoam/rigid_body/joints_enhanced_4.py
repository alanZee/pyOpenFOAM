"""
Enhanced joint types v4 for multi-body rigid body dynamics.

Extends :class:`~pyfoam.rigid_body.joints_enhanced_3` with:

- :class:`ElasticJoint` — compliant joint with linear + angular stiffness (6 DOF)
- :class:`ElectricalJoint` — electromechanical coupling with torque from current (1 DOF)
- :class:`TelescopicJoint` — prismatic with variable stroke limit (1 DOF)
- :class:`PassiveJoint` — free rotation with gravity-driven restoring (1 DOF)

Usage::

    joint = ElasticJoint(
        linear_stiffness=torch.tensor([1e3, 1e3, 1e3]),
        angular_stiffness=torch.tensor([100, 100, 100]),
    )
    assert joint.n_dof == 6

References
----------
- OpenFOAM ``rigidBodyMeshMotion`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.joints import Joint

__all__ = [
    "ElasticJoint",
    "ElectricalJoint",
    "TelescopicJoint",
    "PassiveJoint",
]


class ElasticJoint(Joint):
    """Elastic joint: 6 DOF with linear and angular compliance.

    Provides restoring forces and torques proportional to
    displacement and rotation from the reference configuration::

        F = -K_x * delta_x
        T = -K_theta * delta_theta

    Args:
        linear_stiffness: ``(3,)`` translational stiffness (N/m).
        angular_stiffness: ``(3,)`` rotational stiffness (N*m/rad).
        linear_damping: ``(3,)`` translational damping (N*s/m).
        angular_damping: ``(3,)`` rotational damping (N*m*s/rad).
    """

    def __init__(
        self,
        linear_stiffness: torch.Tensor | None = None,
        angular_stiffness: torch.Tensor | None = None,
        linear_damping: torch.Tensor | None = None,
        angular_damping: torch.Tensor | None = None,
    ) -> None:
        self._k_x = (
            linear_stiffness.to(dtype=torch.float64)
            if linear_stiffness is not None
            else torch.tensor([1e3, 1e3, 1e3], dtype=torch.float64)
        )
        self._k_theta = (
            angular_stiffness.to(dtype=torch.float64)
            if angular_stiffness is not None
            else torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        )
        self._c_x = (
            linear_damping.to(dtype=torch.float64)
            if linear_damping is not None
            else torch.zeros(3, dtype=torch.float64)
        )
        self._c_theta = (
            angular_damping.to(dtype=torch.float64)
            if angular_damping is not None
            else torch.zeros(3, dtype=torch.float64)
        )

    @property
    def n_dof(self) -> int:
        return 6

    def allowed_axes(self) -> torch.Tensor:
        return torch.eye(3, dtype=torch.float64)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """All translation allowed."""
        return dvel

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """All rotation allowed."""
        return domega

    def restoring_force(
        self,
        displacement: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute elastic restoring force.

        Args:
            displacement: ``(3,)`` displacement from reference.
            velocity: ``(3,)`` translational velocity.

        Returns:
            ``(3,)`` restoring force.
        """
        d = displacement.to(dtype=torch.float64)
        v = velocity.to(dtype=torch.float64)
        return -self._k_x * d - self._c_x * v

    def restoring_torque(
        self,
        angle: torch.Tensor,
        angular_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute elastic restoring torque.

        Args:
            angle: ``(3,)`` rotation from reference (rad).
            angular_velocity: ``(3,)`` angular velocity.

        Returns:
            ``(3,)`` restoring torque.
        """
        a = angle.to(dtype=torch.float64)
        w = angular_velocity.to(dtype=torch.float64)
        return -self._k_theta * a - self._c_theta * w


class ElectricalJoint(Joint):
    """Electromechanical joint: torque from electrical current (1 DOF).

    Models a DC motor joint where the torque is proportional to the
    supplied current::

        T = K_t * I

    where K_t is the motor torque constant.

    Args:
        axis: ``(3,)`` rotation axis (will be normalised).
        torque_constant: Motor torque constant K_t (N*m/A).
        resistance: Motor winding resistance (ohm).
        back_emf_constant: Back-EMF constant (V*s/rad).
    """

    def __init__(
        self,
        axis: torch.Tensor,
        torque_constant: float = 0.1,
        resistance: float = 1.0,
        back_emf_constant: float = 0.1,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._K_t = torque_constant
        self._R = resistance
        self._K_e = back_emf_constant

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return domega.dot(self._axis) * self._axis

    def motor_torque(
        self,
        voltage: float,
        angular_velocity: float,
    ) -> float:
        """Compute motor torque from applied voltage.

        Args:
            voltage: Applied voltage (V).
            angular_velocity: Motor speed (rad/s).

        Returns:
            Motor torque (N*m).
        """
        back_emf = self._K_e * angular_velocity
        current = (voltage - back_emf) / max(self._R, 1e-15)
        return self._K_t * current


class TelescopicJoint(Joint):
    """Telescopic joint: prismatic with variable stroke limit (1 DOF).

    Allows translation along an axis with position-dependent limits
    that model a telescoping mechanism.

    Args:
        axis: ``(3,)`` translation axis (will be normalised).
        min_stroke: Minimum extension (m, negative = retracted).
        max_stroke: Maximum extension (m).
        stiffness_beyond: Stiffness when exceeding limits (N/m).
    """

    def __init__(
        self,
        axis: torch.Tensor,
        min_stroke: float = -0.5,
        max_stroke: float = 0.5,
        stiffness_beyond: float = 1e6,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Translation axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._min = min_stroke
        self._max = max_stroke
        self._k_limit = stiffness_beyond

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def limit_force(self, extension: float) -> float:
        """Compute restoring force when exceeding stroke limits.

        Args:
            extension: Current extension along the axis (m).

        Returns:
            Limit force (N, positive = pushing back to range).
        """
        if extension < self._min:
            return self._k_limit * (self._min - extension)
        elif extension > self._max:
            return self._k_limit * (self._max - extension)
        return 0.0

    @property
    def stroke_range(self) -> float:
        """Available stroke range (m)."""
        return self._max - self._min


class PassiveJoint(Joint):
    """Passive revolute joint with gravity-driven restoring torque (1 DOF).

    Models a passive hinge that swings under gravity with a restoring
    torque from its own weight::

        T = -m * g * L * sin(theta)

    Args:
        axis: ``(3,)`` rotation axis (will be normalised).
        mass: Mass of the attached body (kg).
        com_distance: Distance from joint to centre of mass (m).
        gravity_magnitude: Gravitational acceleration magnitude (m/s^2).
        damping: Passive damping coefficient (N*m*s/rad).
    """

    def __init__(
        self,
        axis: torch.Tensor,
        mass: float = 1.0,
        com_distance: float = 0.5,
        gravity_magnitude: float = 9.81,
        damping: float = 0.1,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._mass = mass
        self._L = com_distance
        self._g = gravity_magnitude
        self._c = damping

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return domega.dot(self._axis) * self._axis

    def gravity_torque(
        self,
        angle: float,
        angular_velocity: float,
    ) -> float:
        """Compute gravity-driven restoring torque.

        Args:
            angle: Joint angle from vertical (rad).
            angular_velocity: Angular velocity (rad/s).

        Returns:
            Restoring torque (N*m).
        """
        return (
            -self._mass * self._g * self._L * math.sin(angle)
            - self._c * angular_velocity
        )
