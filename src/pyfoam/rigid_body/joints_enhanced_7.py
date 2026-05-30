"""
Enhanced joint types v7 for multi-body rigid body dynamics.

Extends :class:`~pyfoam.rigid_body.joints_enhanced_6` with:

- :class:`ShapeMemoryAlloyJoint` -- SMA actuator with phase transformation (1 DOF)
- :class:`HydraulicJoint` -- hydraulic cylinder actuator (1 DOF)
- :class:`SuperelasticJoint` -- superelastic NiTi joint with hysteresis (1 DOF)
- :class:`TendonDrivenJoint` -- tendon-driven joint with routing (1 DOF)

Usage::

    sma = ShapeMemoryAlloyJoint(
        axis=torch.tensor([0, 0, 1], dtype=torch.float64),
        austenite_finish_temp=353.15,
    )
    sma.set_temperature(373.15)
    force = sma.actuator_force()

References
----------
- OpenFOAM ``rigidBodyMeshMotion`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.joints import Joint

__all__ = [
    "ShapeMemoryAlloyJoint",
    "HydraulicJoint",
    "SuperelasticJoint",
    "TendonDrivenJoint",
]


class ShapeMemoryAlloyJoint(Joint):
    """形状记忆合金（SMA）驱动关节：温度驱动的相变驱动器（1 DOF）。

    模型::

        F = sigma_martensite * A_wire * xi(T)

    其中 xi(T) 是马氏体体积分数（温度的函数），
    sigma_martensite 是马氏体应力，A_wire 是丝材截面积。

    Args:
        axis: ``(3,)`` 驱动轴（将被归一化）。
        wire_diameter: SMA 丝直径 (m)。
        martensite_stress: 马氏体应力 (Pa)。
        austenite_start_temp: 奥氏体开始温度 As (K)。
        austenite_finish_temp: 奥氏体结束温度 Af (K)。
        max_strain: 最大可恢复应变。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        wire_diameter: float = 1e-3,
        martensite_stress: float = 200e6,
        austenite_start_temp: float = 323.15,
        austenite_finish_temp: float = 353.15,
        max_strain: float = 0.04,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._d = wire_diameter
        self._sigma_m = martensite_stress
        self._As = austenite_start_temp
        self._Af = austenite_finish_temp
        self._eps_max = max_strain
        self._T_current: float = 293.15  # 室温（马氏体状态）
        self._area = math.pi * wire_diameter ** 2 / 4.0

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_temperature(self, temperature: float) -> None:
        """设置当前温度。

        Args:
            temperature: 当前温度 (K)。
        """
        self._T_current = temperature

    @property
    def martensite_fraction(self) -> float:
        """马氏体体积分数（简化线性模型）。"""
        if self._T_current <= self._As:
            return 1.0
        if self._T_current >= self._Af:
            return 0.0
        return 1.0 - (self._T_current - self._As) / (self._Af - self._As)

    def actuator_force(self) -> float:
        """计算 SMA 驱动力。

        Returns:
            驱动力 (N)。
        """
        xi = self.martensite_fraction
        # F = sigma * A * eps_max * xi
        return self._sigma_m * self._area * self._eps_max * xi

    @property
    def current_strain(self) -> float:
        """当前应变。"""
        return self._eps_max * self.martensite_fraction

    @property
    def temperature(self) -> float:
        """当前温度。"""
        return self._T_current


class HydraulicJoint(Joint):
    """液压缸驱动关节：液压驱动的线性执行器（1 DOF）。

    模型::

        F = P * A_piston - P_back * A_annulus

    其中 P 是供油压力，A_piston 是活塞面积，P_back 是回油压力。

    Args:
        axis: ``(3,)`` 驱动轴（将被归一化）。
        piston_diameter: 活塞直径 (m)。
        rod_diameter: 活塞杆直径 (m)。
        max_pressure: 最大供油压力 (Pa)。
        max_stroke: 最大行程 (m)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        piston_diameter: float = 0.05,
        rod_diameter: float = 0.025,
        max_pressure: float = 21e6,
        max_stroke: float = 0.5,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._D_p = piston_diameter
        self._D_r = rod_diameter
        self._P_max = max_pressure
        self._stroke_max = max_stroke
        self._A_piston = math.pi * piston_diameter ** 2 / 4.0
        self._A_annulus = math.pi * (piston_diameter ** 2 - rod_diameter ** 2) / 4.0
        self._P_supply: float = 0.0
        self._P_back: float = 0.0
        self._position: float = 0.0

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_pressure(self, supply: float, back: float = 0.0) -> None:
        """设置液压压力。

        Args:
            supply: 供油压力 (Pa)。
            back: 回油压力 (Pa)。
        """
        self._P_supply = max(0.0, min(self._P_max, supply))
        self._P_back = max(0.0, back)

    def actuator_force(self) -> float:
        """计算液压驱动力。

        Returns:
            驱动力 (N)。
        """
        return self._P_supply * self._A_piston - self._P_back * self._A_annulus

    @property
    def piston_area(self) -> float:
        """活塞面积。"""
        return self._A_piston

    @property
    def annulus_area(self) -> float:
        """环形面积。"""
        return self._A_annulus


class SuperelasticJoint(Joint):
    """超弹性 NiTi 关节：具有滞回行为的超弹性驱动（1 DOF）。

    模型::

        加载: sigma = sigma_s + (sigma_f - sigma_s) * (eps - eps_s) / (eps_f - eps_s)
        卸载: sigma = sigma_s * (eps - eps_r) / (eps_s - eps_r)

    Args:
        axis: ``(3,)`` 关节轴（将被归一化）。
        transformation_start_stress: 相变开始应力 (Pa)。
        transformation_end_stress: 相变结束应力 (Pa)。
        recoverable_strain: 可恢复应变。
        stiffness: 弹性刚度 (N/m)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        transformation_start_stress: float = 400e6,
        transformation_end_stress: float = 600e6,
        recoverable_strain: float = 0.06,
        stiffness: float = 1e6,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._sigma_s = transformation_start_stress
        self._sigma_f = transformation_end_stress
        self._eps_max = recoverable_strain
        self._k = stiffness
        self._current_strain: float = 0.0
        self._is_loading: bool = True
        self._max_strain_reached: float = 0.0

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return domega.dot(self._axis) * self._axis

    def update_strain(self, d_strain: float) -> float:
        """更新应变。

        Args:
            d_strain: 应变增量。

        Returns:
            当前应变。
        """
        self._current_strain += d_strain
        self._max_strain_reached = max(self._max_strain_reached, abs(self._current_strain))
        self._is_loading = d_strain >= 0
        return self._current_strain

    def stress(self) -> float:
        """计算当前应力（简化滞回模型）。"""
        eps = abs(self._current_strain)
        if eps <= 0:
            return 0.0

        if self._is_loading:
            # 加载路径
            if eps < self._sigma_s / self._k:
                return self._k * eps
            else:
                # 相变区域
                phase_frac = min(1.0, eps / max(self._eps_max, 1e-30))
                return self._sigma_s + (self._sigma_f - self._sigma_s) * phase_frac
        else:
            # 卸载路径（简化：线性回到原点）
            eps_r = self._max_strain_reached
            if eps_r <= 0:
                return 0.0
            return self._sigma_s * eps / max(eps_r, 1e-30)

    @property
    def current_strain(self) -> float:
        """当前应变。"""
        return self._current_strain

    @property
    def is_loading(self) -> bool:
        """是否处于加载状态。"""
        return self._is_loading


class TendonDrivenJoint(Joint):
    """腱驱动关节：通过绳索/腱传递力矩的关节（1 DOF）。

    模型::

        tau = r * (F_actuator - F_antagonist)

    其中 r 是腱的力臂，F_actuator 是主动腱力，F_antagonist 是拮抗腱力。

    Args:
        axis: ``(3,)`` 关节轴（将被归一化）。
        moment_arm: 力臂 (m)。
        tendon_stiffness: 腱刚度 (N/m)。
        tendon_damping: 腱阻尼 (N*s/m)。
        pretension: 预张力 (N)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        moment_arm: float = 0.03,
        tendon_stiffness: float = 1e4,
        tendon_damping: float = 10.0,
        pretension: float = 5.0,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._r = moment_arm
        self._k = tendon_stiffness
        self._c = tendon_damping
        self._F_pretension = pretension
        self._actuator_displacement: float = 0.0
        self._antagonist_displacement: float = 0.0

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(dvel)

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return domega.dot(self._axis) * self._axis

    def set_actuator_displacement(self, displacement: float) -> None:
        """设置主动腱位移。

        Args:
            displacement: 主动腱位移 (m)。
        """
        self._actuator_displacement = displacement

    def set_antagonist_displacement(self, displacement: float) -> None:
        """设置拮抗腱位移。

        Args:
            displacement: 拮抗腱位移 (m)。
        """
        self._antagonist_displacement = displacement

    def actuator_tendon_force(self) -> float:
        """计算主动腱力。"""
        return self._F_pretension + self._k * self._actuator_displacement

    def antagonist_tendon_force(self) -> float:
        """计算拮抗腱力。"""
        return self._F_pretension + self._k * self._antagonist_displacement

    def torque(self) -> float:
        """计算关节力矩。

        Returns:
            关节力矩 (N*m)。
        """
        F_act = self.actuator_tendon_force()
        F_ant = self.antagonist_tendon_force()
        return self._r * (F_act - F_ant)

    @property
    def moment_arm(self) -> float:
        """力臂。"""
        return self._r
