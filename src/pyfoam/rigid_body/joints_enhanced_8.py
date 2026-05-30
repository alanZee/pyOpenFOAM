"""
Enhanced joint types v8 for multi-body rigid body dynamics.

Extends :class:`~pyfoam.rigid_body.joints_enhanced_7` with:

- :class:`MagnetostrictiveJoint` -- magnetostrictive actuator with field-dependent strain (1 DOF)
- :class:`ElectroactivePolymerJoint` -- EAP actuator with electric field activation (1 DOF)
- :class:`RotaryLinearJoint` -- rotary-to-linear conversion joint (2 DOF)
- :class:`GearedHarmonicJoint` -- geared harmonic drive with flexspline compliance (1 DOF)

Usage::

    ms = MagnetostrictiveJoint(
        axis=torch.tensor([0, 0, 1], dtype=torch.float64),
        max_strain=1000e-6,
    )
    ms.set_magnetic_field(80e3)
    force = ms.actuator_force()

References
----------
- OpenFOAM ``rigidBodyMeshMotion`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.joints import Joint

__all__ = [
    "MagnetostrictiveJoint",
    "ElectroactivePolymerJoint",
    "RotaryLinearJoint",
    "GearedHarmonicJoint",
]


class MagnetostrictiveJoint(Joint):
    """磁致伸缩驱动关节：磁场驱动的应变执行器（1 DOF）。

    模型::

        epsilon = d * H + s * sigma

    其中 d 是磁致伸缩系数，H 是磁场强度，
    s 是柔度，sigma 是应力。

    Args:
        axis: ``(3,)`` 驱动轴（将被归一化）。
        max_strain: 最大可恢复应变。
        magnetic_coefficient: 磁致伸缩系数 d (m/A)。
        compliance: 柔度 s (1/Pa)。
        prestress: 预应力 (Pa)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        max_strain: float = 1000e-6,
        magnetic_coefficient: float = 1e-9,
        compliance: float = 1e-11,
        prestress: float = -10e6,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._eps_max = max_strain
        self._d = magnetic_coefficient
        self._s = compliance
        self._sigma_0 = prestress
        self._H: float = 0.0  # 磁场强度 (A/m)

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_magnetic_field(self, H: float) -> None:
        """设置磁场强度。

        Args:
            H: 磁场强度 (A/m)。
        """
        self._H = H

    @property
    def current_strain(self) -> float:
        """当前应变。"""
        eps = self._d * self._H + self._s * self._sigma_0
        return max(-self._eps_max, min(self._eps_max, eps))

    def actuator_force(self) -> float:
        """计算驱动力。

        Returns:
            驱动力 (N)。
        """
        return self.current_strain / max(self._s, 1e-30)

    @property
    def magnetic_field(self) -> float:
        """当前磁场强度。"""
        return self._H


class ElectroactivePolymerJoint(Joint):
    """电活性聚合物（EAP）驱动关节：电场驱动的柔性执行器（1 DOF）。

    模型::

        epsilon = epsilon_r * (E / E_breakdown)^2

    其中 epsilon_r 是最大应变，E 是电场强度，
    E_breakdown 是击穿电场。

    Args:
        axis: ``(3,)`` 驱动轴（将被归一化）。
        max_strain: 最大应变。
        breakdown_field: 击穿电场 (V/m)。
        electrode_gap: 电极间距 (m)。
        polymer_stiffness: 聚合物刚度 (Pa)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        max_strain: float = 0.3,
        breakdown_field: float = 200e6,
        electrode_gap: float = 1e-3,
        polymer_stiffness: float = 1e6,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._eps_r = max_strain
        self._E_bd = breakdown_field
        self._gap = electrode_gap
        self._k = polymer_stiffness
        self._V: float = 0.0  # 施加电压 (V)

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
        """设置施加电压。

        Args:
            voltage: 电压 (V)。
        """
        self._V = max(0.0, voltage)

    @property
    def electric_field(self) -> float:
        """电场强度 (V/m)。"""
        return self._V / max(self._gap, 1e-30)

    @property
    def current_strain(self) -> float:
        """当前应变。"""
        E_ratio = self.electric_field / max(self._E_bd, 1e-30)
        return self._eps_r * min(E_ratio ** 2, 1.0)

    def actuator_force(self) -> float:
        """计算驱动力。

        Returns:
            驱动力 (N)。
        """
        return self._k * self.current_strain

    @property
    def is_near_breakdown(self) -> bool:
        """是否接近击穿电场。"""
        return self.electric_field > 0.8 * self._E_bd


class RotaryLinearJoint(Joint):
    """旋转-直线转换关节：将旋转运动转换为直线运动（2 DOF）。

    模型::

        x = r * theta
        F_linear = tau / r

    其中 r 是丝杠半径，theta 是旋转角度。

    Args:
        rotation_axis: ``(3,)`` 旋转轴。
        linear_axis: ``(3,)`` 直线轴。
        lead_screw_radius: 丝杠半径 (m)。
        lead_screw_pitch: 丝杠螺距 (m/rev)。
        efficiency: 传动效率。
    """

    def __init__(
        self,
        rotation_axis: torch.Tensor,
        linear_axis: torch.Tensor,
        lead_screw_radius: float = 0.01,
        lead_screw_pitch: float = 0.005,
        efficiency: float = 0.9,
    ) -> None:
        for ax, name in [(rotation_axis, "rotation"), (linear_axis, "linear")]:
            norm = ax.norm()
            if norm < 1e-12:
                raise ValueError(f"{name} axis must be non-zero.")

        self._rot_axis = rotation_axis.to(dtype=torch.float64) / rotation_axis.norm()
        self._lin_axis = linear_axis.to(dtype=torch.float64) / linear_axis.norm()
        self._r = lead_screw_radius
        self._pitch = lead_screw_pitch
        self._eta = efficiency

    @property
    def n_dof(self) -> int:
        return 2

    def allowed_axes(self) -> torch.Tensor:
        return torch.stack([self._rot_axis, self._lin_axis])

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._lin_axis) * self._lin_axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return domega.dot(self._rot_axis) * self._rot_axis

    def linear_from_rotation(self, angle: float) -> float:
        """旋转角度转换为直线位移。

        Args:
            angle: 旋转角度 (rad)。

        Returns:
            直线位移 (m)。
        """
        return self._pitch * angle / (2.0 * math.pi)

    def force_from_torque(self, torque: float) -> float:
        """力矩转换为直线力。

        Args:
            torque: 输入力矩 (N*m)。

        Returns:
            输出直线力 (N)。
        """
        return self._eta * torque / max(self._r, 1e-30)

    @property
    def mechanical_advantage(self) -> float:
        """机械增益。"""
        return self._eta / max(self._r, 1e-30)


class GearedHarmonicJoint(Joint):
    """齿轮谐波驱动关节：具有柔轮柔性的谐波驱动（1 DOF）。

    模型::

        theta_out = theta_in / N
        tau_out = N * tau_in * eta

    其中 N 是减速比，eta 是效率。考虑柔轮柔性::

        k_flexspline = k_base * (1 - strain_ratio)

    Args:
        axis: ``(3,)`` 关节轴。
        gear_ratio: 减速比。
        efficiency: 传动效率。
        flexspline_stiffness: 柔轮刚度 (N*m/rad)。
        flexspline_strain_limit: 柔轮应变极限。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        gear_ratio: float = 100,
        efficiency: float = 0.85,
        flexspline_stiffness: float = 1e4,
        flexspline_strain_limit: float = 0.01,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._N = gear_ratio
        self._eta = efficiency
        self._k_flex = flexspline_stiffness
        self._eps_limit = flexspline_strain_limit
        self._input_angle: float = 0.0
        self._current_strain: float = 0.0

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return domega.dot(self._axis) * self._axis

    def set_input_angle(self, angle: float) -> None:
        """设置输入角度。

        Args:
            angle: 输入角度 (rad)。
        """
        self._input_angle = angle
        # 简化应变模型
        self._current_strain = abs(angle) / max(self._N, 1) * 0.001
        self._current_strain = min(self._current_strain, self._eps_limit)

    @property
    def output_angle(self) -> float:
        """输出角度。"""
        return self._input_angle / max(self._N, 1)

    @property
    def effective_stiffness(self) -> float:
        """有效刚度（考虑柔轮柔性）。"""
        return self._k_flex * (1.0 - self._current_strain / max(self._eps_limit, 1e-30))

    def output_torque(self, input_torque: float) -> float:
        """计算输出力矩。

        Args:
            input_torque: 输入力矩 (N*m)。

        Returns:
            输出力矩 (N*m)。
        """
        return self._N * input_torque * self._eta

    @property
    def strain_ratio(self) -> float:
        """当前应变比。"""
        return self._current_strain / max(self._eps_limit, 1e-30)
