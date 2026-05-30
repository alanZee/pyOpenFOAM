"""
Enhanced joint types v6 for multi-body rigid body dynamics.

Extends :class:`~pyfoam.rigid_body.joints_enhanced_5` with:

- :class:`PiezoelectricJoint` -- piezoelectric actuator with voltage control (1 DOF)
- :class:`VariableStiffnessJoint` -- adjustable stiffness via antagonistic actuation (1 DOF)
- :class:`FrictionJoint` -- Stribeck friction model joint (1 DOF)
- :class:`MagneticLevitationJoint` -- magnetic levitation with force feedback (1 DOF)

Usage::

    joint = PiezoelectricJoint(
        axis=torch.tensor([0, 0, 1], dtype=torch.float64),
        piezo_constant=1e-6,
    )
    joint.set_voltage(100.0)
    force = joint.actuator_force()

References
----------
- OpenFOAM ``rigidBodyMeshMotion`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.joints import Joint

__all__ = [
    "PiezoelectricJoint",
    "VariableStiffnessJoint",
    "FrictionJoint",
    "MagneticLevitationJoint",
]


class PiezoelectricJoint(Joint):
    """压电驱动关节：电压控制的纳米级精密驱动器（1 DOF）。

    模型::

        F = d * V / t

    其中 d 是压电常数 (C/N 或 m/V)，V 是施加电压，t 是压电片厚度。

    Args:
        axis: ``(3,)`` 驱动轴（将被归一化）。
        piezo_constant: 压电常数 d (m/V)。
        thickness: 压电片厚度 (m)。
        max_voltage: 最大允许电压 (V)。
        stiffness: 开环刚度 (N/m)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        piezo_constant: float = 1e-6,
        thickness: float = 1e-3,
        max_voltage: float = 1000.0,
        stiffness: float = 1e6,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._d = piezo_constant
        self._t = thickness
        self._V_max = max_voltage
        self._k = stiffness
        self._voltage: float = 0.0

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_voltage(self, voltage: float) -> None:
        """设置驱动电压。

        Args:
            voltage: 施加电压 (V)。
        """
        self._voltage = max(-self._V_max, min(self._V_max, voltage))

    def actuator_force(self) -> float:
        """计算压电驱动力。

        Returns:
            驱动力 (N)。
        """
        # F = d * V / t * k (简化模型)
        displacement = self._d * self._voltage
        return self._k * displacement

    @property
    def current_displacement(self) -> float:
        """当前位移 (m)。"""
        return self._d * self._voltage

    @property
    def voltage(self) -> float:
        """当前电压 (V)。"""
        return self._voltage


class VariableStiffnessJoint(Joint):
    """可变刚度关节：通过拮抗调节实现刚度可调（1 DOF）。

    模型::

        k_eff = k1 + k2 + 2*sqrt(k1*k2)*cos(theta)

    其中 theta 是拮抗角，通过改变拮抗角调节有效刚度。

    Args:
        axis: ``(3,)`` 旋转轴（将被归一化）。
        stiffness_1: 第一根弹簧刚度 (N*m/rad)。
        stiffness_2: 第二根弹簧刚度 (N*m/rad)。
        antagonist_angle_range: 拮抗角范围 [min, max] (rad)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        stiffness_1: float = 100.0,
        stiffness_2: float = 100.0,
        antagonist_angle_range: tuple[float, float] = (0.0, math.pi),
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._k1 = stiffness_1
        self._k2 = stiffness_2
        self._theta_range = antagonist_angle_range
        self._antagonist_angle: float = antagonist_angle_range[0]

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return domega.dot(self._axis) * self._axis

    def set_antagonist_angle(self, angle: float) -> None:
        """设置拮抗角。

        Args:
            angle: 拮抗角 (rad)。
        """
        lo, hi = self._theta_range
        self._antagonist_angle = max(lo, min(hi, angle))

    @property
    def effective_stiffness(self) -> float:
        """计算当前有效刚度。

        Returns:
            有效刚度 (N*m/rad)。
        """
        theta = self._antagonist_angle
        cross_term = 2.0 * math.sqrt(self._k1 * self._k2) * math.cos(theta)
        return max(0.0, self._k1 + self._k2 + cross_term)

    def set_stiffness(self, target_k: float) -> None:
        """设置目标刚度（自动计算所需拮抗角）。

        Args:
            target_k: 目标有效刚度 (N*m/rad)。
        """
        # k_eff = k1 + k2 + 2*sqrt(k1*k2)*cos(theta)
        # cos(theta) = (k_eff - k1 - k2) / (2*sqrt(k1*k2))
        k_sum = self._k1 + self._k2
        k_cross = 2.0 * math.sqrt(self._k1 * self._k2)

        if abs(k_cross) < 1e-30:
            return

        cos_theta = (target_k - k_sum) / k_cross
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)
        self.set_antagonist_angle(theta)


class FrictionJoint(Joint):
    """Stribeck 摩擦关节：包含静摩擦和 Stribeck 效应的摩擦模型（1 DOF）。

    模型::

        F = F_c * sign(v) + (F_s - F_c) * exp(-(v/v_s)^2) * sign(v) + b*v

    其中 F_c 是库仑摩擦，F_s 是静摩擦，v_s 是 Stribeck 速度，b 是粘性系数。

    Args:
        axis: ``(3,)`` 运动轴（将被归一化）。
        coulomb_friction: 库仑摩擦力 (N)。
        static_friction: 最大静摩擦力 (N)。
        stribeck_velocity: Stribeck 速度 (m/s)。
        viscous_coefficient: 粘性摩擦系数 (N*s/m)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        coulomb_friction: float = 10.0,
        static_friction: float = 15.0,
        stribeck_velocity: float = 0.01,
        viscous_coefficient: float = 1.0,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._F_c = coulomb_friction
        self._F_s = static_friction
        self._v_s = stribeck_velocity
        self._b = viscous_coefficient

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return domega.dot(self._axis) * self._axis

    def friction_force(self, velocity: float) -> float:
        """计算 Stribeck 摩擦力。

        Args:
            velocity: 沿轴方向的速度 (m/s 或 rad/s)。

        Returns:
            摩擦力 (N 或 N*m)，与运动方向相反。
        """
        v = abs(velocity)
        if v < 1e-15:
            return 0.0

        sign_v = 1.0 if velocity > 0 else -1.0

        # Stribeck 曲线
        stribeck = (self._F_s - self._F_c) * math.exp(-(v / max(self._v_s, 1e-15)) ** 2)

        # 总摩擦力
        f_total = self._F_c + stribeck + self._b * v

        return -f_total * sign_v

    @property
    def stribeck_velocity(self) -> float:
        """Stribeck 速度。"""
        return self._v_s


class MagneticLevitationJoint(Joint):
    """磁悬浮关节：电磁力驱动的无接触支撑（1 DOF）。

    模型::

        F = k_mag / gap^2 - m*g

    其中 k_mag 是磁力常数，gap 是悬浮间隙，m 是质量。

    Args:
        axis: ``(3,)`` 悬浮轴（将被归一化）。
        magnetic_constant: 磁力常数 k_mag (N*m^2)。
        nominal_gap: 标称悬浮间隙 (m)。
        body_mass: 被悬浮体的质量 (kg)。
        feedback_gain: 间隙反馈增益。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        magnetic_constant: float = 1e-4,
        nominal_gap: float = 0.01,
        body_mass: float = 1.0,
        feedback_gain: float = 1e3,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._k_mag = magnetic_constant
        self._gap_nom = nominal_gap
        self._mass = body_mass
        self._gain = feedback_gain
        self._current_gap: float = nominal_gap

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def update_gap(self, gap: float) -> None:
        """更新当前悬浮间隙。

        Args:
            gap: 当前间隙 (m)。
        """
        self._current_gap = max(gap, 1e-6)

    def levitation_force(self) -> float:
        """计算磁悬浮力。

        Returns:
            悬浮力 (N)，正值向上。
        """
        # 磁吸引力（反比于间隙平方）
        gap = max(self._current_gap, 1e-6)
        magnetic = self._k_mag / (gap * gap)

        # 重力补偿
        gravity_compensation = self._mass * 9.81

        # PD 反馈控制
        gap_error = self._current_gap - self._gap_nom
        feedback = -self._gain * gap_error

        return magnetic - gravity_compensation + feedback

    @property
    def nominal_gap(self) -> float:
        """标称悬浮间隙。"""
        return self._gap_nom
