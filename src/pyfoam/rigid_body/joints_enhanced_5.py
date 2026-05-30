"""
Enhanced joint types v5 for multi-body rigid body dynamics.

Extends :class:`~pyfoam.rigid_body.joints_enhanced_4` with:

- :class:`MagnetorheologicalJoint` -- controllable damping via MR fluid (1 DOF)
- :class:`PneumaticJoint` -- pneumatic cylinder actuator (1 DOF)
- :class:`HarmonicDriveJoint` -- high-ratio harmonic gear drive (1 DOF)
- :class:`RollingContactJoint` -- rolling contact constraint (1 DOF)

Usage::

    joint = MagnetorheologicalJoint(
        axis=torch.tensor([0, 0, 1], dtype=torch.float64),
        max_damping=1000.0,
    )
    joint.set_field_strength(0.5)

References
----------
- OpenFOAM ``rigidBodyMeshMotion`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.joints import Joint

__all__ = [
    "MagnetorheologicalJoint",
    "PneumaticJoint",
    "HarmonicDriveJoint",
    "RollingContactJoint",
]


class MagnetorheologicalJoint(Joint):
    """Magnetorheological (MR) damper joint: controllable damping (1 DOF).

    Models a joint with magnetorheological fluid damper where the
    damping coefficient is controlled by an applied magnetic field::

        T_damp = -c(B) * omega

    where c(B) is the field-dependent damping coefficient.

    Args:
        axis: ``(3,)`` rotation axis (will be normalised).
        min_damping: Minimum damping (zero field) (N*m*s/rad).
        max_damping: Maximum damping (full field) (N*m*s/rad).
        response_time: MR fluid response time constant (s).
    """

    def __init__(
        self,
        axis: torch.Tensor,
        min_damping: float = 1.0,
        max_damping: float = 500.0,
        response_time: float = 0.01,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._c_min = min_damping
        self._c_max = max_damping
        self._tau = response_time
        self._field_strength: float = 0.0  # 0 = off, 1 = full field
        self._effective_damping: float = min_damping

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return domega.dot(self._axis) * self._axis

    def set_field_strength(self, strength: float) -> None:
        """Set the magnetic field strength (0 to 1).

        Args:
            strength: Normalised field strength.
        """
        self._field_strength = max(0.0, min(1.0, strength))
        # 更新有效阻尼系数
        self._effective_damping = (
            self._c_min
            + (self._c_max - self._c_min) * self._field_strength
        )

    def damping_torque(self, angular_velocity: float) -> float:
        """Compute damping torque.

        Args:
            angular_velocity: Angular velocity (rad/s).

        Returns:
            Damping torque (N*m, opposing motion).
        """
        return -self._effective_damping * angular_velocity

    @property
    def current_damping(self) -> float:
        """Current effective damping coefficient."""
        return self._effective_damping


class PneumaticJoint(Joint):
    """Pneumatic cylinder joint: linear actuator (1 DOF).

    Models a pneumatic cylinder with pressure-dependent force::

        F = (p1*A1 - p2*A2) - friction

    where p1, p2 are chamber pressures and A1, A2 are piston areas.

    Args:
        axis: ``(3,)`` translation axis (will be normalised).
        piston_area: Piston area (m^2).
        rod_area: Rod-side area (m^2).
        supply_pressure: Supply pressure (Pa).
        exhaust_pressure: Exhaust pressure (Pa).
    """

    def __init__(
        self,
        axis: torch.Tensor,
        piston_area: float = 1e-3,
        rod_area: float = 5e-4,
        supply_pressure: float = 6e5,
        exhaust_pressure: float = 1e5,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Translation axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._A1 = piston_area
        self._A2 = rod_area
        self._p_s = supply_pressure
        self._p_e = exhaust_pressure

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def actuator_force(self, extend: bool = True) -> float:
        """Compute pneumatic actuator force.

        Args:
            extend: If True, compute extension force; otherwise retraction.

        Returns:
            Actuator force (N).
        """
        if extend:
            return self._p_s * self._A1 - self._p_e * self._A2
        else:
            return self._p_s * self._A2 - self._p_e * self._A1

    @property
    def stroke_force_difference(self) -> float:
        """Difference between extension and retraction forces."""
        return self.actuator_force(extend=True) - self.actuator_force(extend=False)


class HarmonicDriveJoint(Joint):
    """Harmonic drive joint: high-ratio gear transmission (1 DOF).

    Models a harmonic drive with very high reduction ratio and
    near-zero backlash::

        T_output = ratio * T_input
        omega_output = omega_input / ratio

    Args:
        axis: ``(3,)`` rotation axis (will be normalised).
        gear_ratio: Harmonic drive gear ratio (typically 30-320).
        efficiency: Transmission efficiency (0 to 1).
        torsional_stiffness: Output shaft torsional stiffness (N*m/rad).
    """

    def __init__(
        self,
        axis: torch.Tensor,
        gear_ratio: float = 100.0,
        efficiency: float = 0.9,
        torsional_stiffness: float = 1e4,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._ratio = gear_ratio
        self._eta = efficiency
        self._k_t = torsional_stiffness

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return domega.dot(self._axis) * self._axis

    def output_torque(self, input_torque: float) -> float:
        """Compute output torque from input torque.

        Args:
            input_torque: Motor-side torque (N*m).

        Returns:
            Load-side torque (N*m).
        """
        return self._ratio * input_torque * self._eta

    def output_speed(self, input_speed: float) -> float:
        """Compute output speed from input speed.

        Args:
            input_speed: Motor-side speed (rad/s).

        Returns:
            Load-side speed (rad/s).
        """
        return input_speed / self._ratio

    @property
    def gear_ratio(self) -> float:
        """Gear ratio."""
        return self._ratio


class RollingContactJoint(Joint):
    """Rolling contact joint: constraint from rolling without slipping (1 DOF).

    Models the kinematic constraint of a wheel or roller::

        v = omega * R

    where R is the roller radius. Provides a restoring force when
    the no-slip condition is violated.

    Args:
        axis: ``(3,)`` rotation axis (will be normalised).
        roller_radius: Radius of the roller (m).
        contact_stiffness: Penalty stiffness for slip violation (N/m).
    """

    def __init__(
        self,
        axis: torch.Tensor,
        roller_radius: float = 0.1,
        contact_stiffness: float = 1e6,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._R = roller_radius
        self._k = contact_stiffness

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return domega.dot(self._axis) * self._axis

    def rolling_constraint_force(
        self,
        linear_velocity: float,
        angular_velocity: float,
    ) -> float:
        """Compute restoring force for rolling constraint violation.

        Args:
            linear_velocity: Linear velocity along the axis (m/s).
            angular_velocity: Angular velocity about the axis (rad/s).

        Returns:
            Restoring force (N).
        """
        # No-slip: v = omega * R
        slip = linear_velocity - angular_velocity * self._R
        return -self._k * slip

    @property
    def roller_radius(self) -> float:
        """Roller radius."""
        return self._R
