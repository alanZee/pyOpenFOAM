"""
Enhanced restraint types v6 for rigid body motion solvers.

Extends :class:`~pyfoam.rigid_body.restraints_enhanced_5` with:

- :class:`ViscoelasticRestraint` -- Kelvin-Voigt viscoelastic model
- :class:`BistableSpringRestraint` -- bistable spring with snap-through
- :class:`ThermalExpansionRestraint` -- thermal expansion force
- :class:`CreepRestraint` -- Norton-Bailey creep model

Usage::

    visco = ViscoelasticRestraint(
        spring_constant=1e4,
        damping_coefficient=1e2,
    )
    force = visco.force(position, velocity)

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` restraint models
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.restraints import Restraint

__all__ = [
    "ViscoelasticRestraint",
    "BistableSpringRestraint",
    "ThermalExpansionRestraint",
    "CreepRestraint",
]


class ViscoelasticRestraint(Restraint):
    """Kelvin-Voigt 粘弹性约束。

    模型::

        F = -k*x - c*v

    其中 k 是弹簧刚度，c 是阻尼系数，x 是位移，v 是速度。

    Args:
        spring_constant: 弹簧刚度 k (N/m).
        damping_coefficient: 阻尼系数 c (N*s/m).
        rest_position: ``(3,)`` 平衡位置。
    """

    def __init__(
        self,
        spring_constant: float = 1e4,
        damping_coefficient: float = 1e2,
        rest_position: torch.Tensor | None = None,
    ) -> None:
        self._k = spring_constant
        self._c = damping_coefficient
        self._rest = (
            rest_position.to(dtype=torch.float64)
            if rest_position is not None
            else torch.zeros(3, dtype=torch.float64)
        )

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算 Kelvin-Voigt 粘弹性力。

        Args:
            position: ``(3,)`` 物体位置。
            velocity: ``(3,)`` 物体速度。

        Returns:
            ``(3,)`` 约束力。
        """
        pos = position.to(dtype=torch.float64)
        vel = velocity.to(dtype=torch.float64)
        displacement = pos - self._rest
        return -self._k * displacement - self._c * vel

    @property
    def spring_constant(self) -> float:
        """弹簧刚度。"""
        return self._k

    @property
    def damping_coefficient(self) -> float:
        """阻尼系数。"""
        return self._c


class BistableSpringRestraint(Restraint):
    """双稳态弹簧约束：具有 snap-through 行为。

    模型::

        F = -k * x * (x^2 - a^2)

    其中 a 是两个稳定平衡点之间的半距离。力在 x = 0 处为零（不稳定平衡），
    在 x = +/- a 处为零（稳定平衡）。

    Args:
        stiffness: 弹簧刚度 k (N/m^3).
        equilibrium_distance: 稳定平衡点距离 a (m).
        energy_barrier: 能量势垒高度 (J)。
    """

    def __init__(
        self,
        stiffness: float = 1e6,
        equilibrium_distance: float = 0.01,
        energy_barrier: float | None = None,
    ) -> None:
        self._k = stiffness
        self._a = equilibrium_distance
        # 如果指定了能量势垒，从中推导刚度
        if energy_barrier is not None and equilibrium_distance > 0:
            self._k = energy_barrier / (0.25 * equilibrium_distance ** 4)

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算双稳态弹簧力。

        Args:
            position: ``(3,)`` 物体位置。
            velocity: ``(3,)`` 物体速度（未使用）。

        Returns:
            ``(3,)`` 约束力。
        """
        pos = position.to(dtype=torch.float64)
        r = pos.norm()

        if r < 1e-30:
            return torch.zeros(3, dtype=torch.float64)

        direction = pos / r
        # F = -k * r * (r^2 - a^2)
        force_mag = -self._k * r * (r ** 2 - self._a ** 2)
        return force_mag * direction

    @property
    def equilibrium_distance(self) -> float:
        """稳定平衡点距离。"""
        return self._a

    def potential_energy(self, position: torch.Tensor) -> float:
        """计算势能。

        Args:
            position: ``(3,)`` 物体位置。

        Returns:
            势能值 (J)。
        """
        r = position.to(dtype=torch.float64).norm().item()
        # U = k/4 * (r^2 - a^2)^2
        return self._k / 4.0 * (r ** 2 - self._a ** 2) ** 2


class ThermalExpansionRestraint(Restraint):
    """热膨胀约束：温度变化引起的膨胀力。

    模型::

        F = -k * alpha * dT * L0 * direction

    其中 alpha 是热膨胀系数，dT 是温度变化，L0 是参考长度。

    Args:
        stiffness: 结构刚度 (N/m).
        thermal_expansion_coefficient: 热膨胀系数 alpha (1/K).
        reference_length: 参考长度 L0 (m).
        reference_temperature: 参考温度 T0 (K).
    """

    def __init__(
        self,
        stiffness: float = 1e6,
        thermal_expansion_coefficient: float = 12e-6,
        reference_length: float = 1.0,
        reference_temperature: float = 293.15,
    ) -> None:
        self._k = stiffness
        self._alpha = thermal_expansion_coefficient
        self._L0 = reference_length
        self._T_ref = reference_temperature
        self._T_current: float = reference_temperature

    def set_temperature(self, temperature: float) -> None:
        """设置当前温度。

        Args:
            temperature: 当前温度 (K)。
        """
        self._T_current = temperature

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算热膨胀约束力。

        Args:
            position: ``(3,)`` 物体位置。
            velocity: ``(3,)`` 物体速度（未使用）。

        Returns:
            ``(3,)`` 热膨胀力。
        """
        pos = position.to(dtype=torch.float64)
        r = pos.norm()

        if r < 1e-30:
            return torch.zeros(3, dtype=torch.float64)

        direction = pos / r
        dT = self._T_current - self._T_ref

        # 热膨胀引起的应变
        thermal_strain = self._alpha * dT
        thermal_displacement = thermal_strain * self._L0

        # 约束力（阻止膨胀）
        force_mag = -self._k * thermal_displacement
        return force_mag * direction

    @property
    def temperature_change(self) -> float:
        """温度变化量 (K)。"""
        return self._T_current - self._T_ref

    @property
    def thermal_strain(self) -> float:
        """热应变。"""
        return self._alpha * self.temperature_change


class CreepRestraint(Restraint):
    """Norton-Bailey 蠕变约束：高温长时间蠕变行为。

    模型::

        d(eps_c)/dt = A * sigma^n * t^(m-1)

    其中 A 是蠕变常数，n 是应力指数，m 是时间指数。

    Args:
        creep_constant: 蠕变常数 A (1/(Pa^n * s^m)).
        stress_exponent: 应力指数 n.
        time_exponent: 时间指数 m.
        stiffness: 约束刚度 (N/m).
    """

    def __init__(
        self,
        creep_constant: float = 1e-20,
        stress_exponent: float = 5.0,
        time_exponent: float = 0.3,
        stiffness: float = 1e6,
    ) -> None:
        self._A = creep_constant
        self._n = stress_exponent
        self._m = time_exponent
        self._k = stiffness
        self._creep_strain: float = 0.0
        self._time: float = 0.0
        self._dt: float = 0.001  # 默认时间步

    def set_time_step(self, dt: float) -> None:
        """设置时间步长。

        Args:
            dt: 时间步长 (s)。
        """
        self._dt = dt

    def update_creep(
        self,
        stress_magnitude: float,
        dt: float | None = None,
    ) -> float:
        """更新蠕变应变。

        Args:
            stress_magnitude: 应力幅值 (Pa)。
            dt: 时间步长（使用默认值若 None）。

        Returns:
            更新后的蠕变应变。
        """
        actual_dt = dt if dt is not None else self._dt
        self._time += actual_dt

        if stress_magnitude <= 0 or self._time <= 0:
            return self._creep_strain

        # Norton-Bailey: d(eps_c)/dt = A * sigma^n * t^(m-1)
        creep_rate = (
            self._A
            * stress_magnitude ** self._n
            * self._time ** (self._m - 1.0)
        )

        self._creep_strain += creep_rate * actual_dt
        return self._creep_strain

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算蠕变约束力。

        Args:
            position: ``(3,)`` 物体位置。
            velocity: ``(3,)`` 物体速度（未使用）。

        Returns:
            ``(3,)`` 蠕变恢复力。
        """
        pos = position.to(dtype=torch.float64)
        r = pos.norm()

        if r < 1e-30:
            return torch.zeros(3, dtype=torch.float64)

        direction = pos / r
        # 蠕变应变产生的位移
        creep_displacement = self._creep_strain * r
        force_mag = -self._k * creep_displacement
        return force_mag * direction

    @property
    def accumulated_creep_strain(self) -> float:
        """累积蠕变应变。"""
        return self._creep_strain

    def reset(self) -> None:
        """重置蠕变状态。"""
        self._creep_strain = 0.0
        self._time = 0.0
