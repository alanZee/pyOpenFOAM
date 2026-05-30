"""
Enhanced joint types v10 for multi-body rigid body dynamics.

Extends :class:`~pyfoam.rigid_body.joints_enhanced_9` with:

- :class:`MagnetorheologicalCompositeJoint` -- MR composite with field-dependent stiffness (1 DOF)
- :class:`ElectroactiveHydrogelJoint` -- hydrogel actuator with pH/ionic response (1 DOF)
- :class:`DielectricElastomerJoint` -- DE actuator with Maxwell pressure model (1 DOF)
- :class:`ThermoplasticMemoryJoint` -- thermoplastic shape memory with multi-cycle memory (1 DOF)

Usage::

    de = DielectricElastomerJoint(
        axis=torch.tensor([0, 0, 1], dtype=torch.float64),
        electrode_area=1e-3,
    )
    de.set_voltage(5000.0)
    force = de.actuator_force()

References
----------
- OpenFOAM ``rigidBodyMeshMotion`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.joints import Joint

__all__ = [
    "MagnetorheologicalCompositeJoint",
    "ElectroactiveHydrogelJoint",
    "DielectricElastomerJoint",
    "ThermoplasticMemoryJoint",
]


class MagnetorheologicalCompositeJoint(Joint):
    """磁流变复合材料关节：场致刚度变化执行器（1 DOF）。

    模型::

        k_eff = k_base + k_mr * B^n / (B^n + B_half^n)
        F = k_eff * delta_x

    Args:
        axis: ``(3,)`` 驱动轴。
        base_stiffness: 基础刚度 (N/m)。
        mr_stiffness: MR 最大刚度增量 (N/m)。
        half_field: 半刚度磁场 (T)。
        field_exponent: 磁场指数。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        base_stiffness: float = 1e3,
        mr_stiffness: float = 1e4,
        half_field: float = 0.5,
        field_exponent: float = 2.0,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._k_base = base_stiffness
        self._k_mr = mr_stiffness
        self._B_half = half_field
        self._n = field_exponent
        self._B: float = 0.0
        self._displacement: float = 0.0

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_field(self, B: float) -> None:
        """设置磁场强度 (T)。"""
        self._B = max(0.0, B)

    def set_displacement(self, delta: float) -> None:
        """设置位移 (m)。"""
        self._displacement = delta

    @property
    def effective_stiffness(self) -> float:
        """当前有效刚度。"""
        ratio = self._B ** self._n / max(self._B ** self._n + self._B_half ** self._n, 1e-30)
        return self._k_base + self._k_mr * ratio

    def actuator_force(self) -> float:
        """计算输出力。"""
        return self.effective_stiffness * self._displacement


class ElectroactiveHydrogelJoint(Joint):
    """电活性水凝胶关节：pH/离子响应执行器（1 DOF）。

    模型::

        V_swell = V_0 * (1 + alpha * (pH - pH_0))
        F = k_hydrogel * (V_swell - V_0) / A

    Args:
        axis: ``(3,)`` 驱动轴。
        reference_volume: 参考体积 (m^3)。
        swelling_coefficient: 溶胀系数。
        reference_ph: 参考 pH 值。
        hydrogel_stiffness: 水凝胶刚度 (N/m)。
        cross_section_area: 截面积 (m^2)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        reference_volume: float = 1e-6,
        swelling_coefficient: float = 0.5,
        reference_ph: float = 7.0,
        hydrogel_stiffness: float = 1e3,
        cross_section_area: float = 1e-4,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._V0 = reference_volume
        self._alpha = swelling_coefficient
        self._pH0 = reference_ph
        self._k = hydrogel_stiffness
        self._A = cross_section_area
        self._pH: float = reference_ph

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_ph(self, pH: float) -> None:
        """设置 pH 值。"""
        self._pH = max(0.0, min(14.0, pH))

    @property
    def swelling_ratio(self) -> float:
        """溶胀比。"""
        return 1.0 + self._alpha * (self._pH - self._pH0)

    def actuator_force(self) -> float:
        """计算输出力。"""
        V_swell = self._V0 * self.swelling_ratio
        delta_V = V_swell - self._V0
        displacement = delta_V / max(self._A, 1e-30)
        return self._k * displacement


class DielectricElastomerJoint(Joint):
    """介电弹性体关节：Maxwell 压力模型执行器（1 DOF）。

    模型::

        P_maxwell = eps_0 * eps_r * (V / t)^2
        F = P_maxwell * A * strain

    Args:
        axis: ``(3,)`` 驱动轴。
        electrode_area: 电极面积 (m^2)。
        film_thickness: 薄膜厚度 (m)。
        relative_permittivity: 相对介电常数。
        max_strain: 最大应变。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        electrode_area: float = 1e-3,
        film_thickness: float = 50e-6,
        relative_permittivity: float = 3.0,
        max_strain: float = 0.3,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._A = electrode_area
        self._t = film_thickness
        self._eps_r = relative_permittivity
        self._eps_0 = 8.854e-12  # 真空介电常数 (F/m)
        self._max_strain = max_strain
        self._V: float = 0.0

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_voltage(self, V: float) -> None:
        """设置电压 (V)。"""
        self._V = max(0.0, V)

    @property
    def maxwell_pressure(self) -> float:
        """Maxwell 压力 (Pa)。"""
        E_field = self._V / max(self._t, 1e-30)
        return self._eps_0 * self._eps_r * E_field ** 2

    def actuator_force(self) -> float:
        """计算输出力。"""
        P = self.maxwell_pressure
        strain = min(self._max_strain, P / max(1e6, 1e-30))
        return P * self._A * strain


class ThermoplasticMemoryJoint(Joint):
    """热塑性形状记忆关节：多周期记忆执行器（1 DOF）。

    模型::

        eps = eps_mem[T_stage] * H_stage(T)
        H_stage = sum of sigmoid transitions for each stage

    Args:
        axis: ``(3,)`` 驱动轴。
        stage_temperatures: 各阶段转变温度列表 (K)。
        stage_strains: 各阶段记忆应变列表。
        transition_width: 相变温度区间宽度 (K)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        stage_temperatures: list[float] | None = None,
        stage_strains: list[float] | None = None,
        transition_width: float = 10.0,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._T_stages = stage_temperatures or [320.0, 360.0, 400.0]
        self._eps_stages = stage_strains or [0.02, 0.04, 0.06]
        self._a = transition_width
        self._T: float = 293.15

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_temperature(self, T: float) -> None:
        """设置温度 (K)。"""
        self._T = T

    @property
    def current_strain(self) -> float:
        """当前应变（各阶段贡献之和）。"""
        total = 0.0
        for i, (T_s, eps_s) in enumerate(zip(self._T_stages, self._eps_stages)):
            H = 0.5 * (1.0 + math.tanh((self._T - T_s) / max(self._a, 1e-10)))
            total += eps_s * H
        return total

    def actuator_force(self) -> float:
        """计算驱动力。"""
        return self.current_strain * 1e6  # 等效刚度 1 MPa
