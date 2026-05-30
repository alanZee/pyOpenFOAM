"""
Enhanced restraint types v7 for rigid body motion solvers.

Extends :class:`~pyfoam.rigid_body.restraints_enhanced_6` with:

- :class:`MagnetorheologicalRestraint` -- MR fluid damper with field-dependent yield stress
- :class:`FrictionPendulumRestraint` -- friction pendulum system for seismic isolation
- :class:`ParticleDamperRestraint` -- particle damper with granular dynamics
- :class:`NegativeStiffnessRestraint` -- negative stiffness for vibration isolation

Usage::

    mr = MagnetorheologicalRestraint(
        yield_stress_min=0.0,
        yield_stress_max=60e3,
    )
    mr.set_field_strength(0.8)
    force = mr.force(position, velocity)

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` restraint models
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.restraints import Restraint

__all__ = [
    "MagnetorheologicalRestraint",
    "FrictionPendulumRestraint",
    "ParticleDamperRestraint",
    "NegativeStiffnessRestraint",
]


class MagnetorheologicalRestraint(Restraint):
    """磁流变（MR）流体阻尼器：磁场相关的屈服应力阻尼器。

    模型::

        F = -(tau_y(H) * A * sign(v) + c * v)

    其中 tau_y(H) 是与磁场强度相关的屈服应力，
    A 是活塞面积，c 是粘性阻尼系数。

    Args:
        yield_stress_min: 零磁场屈服应力 (Pa)。
        yield_stress_max: 饱和磁场屈服应力 (Pa)。
        piston_area: 活塞面积 (m^2)。
        viscous_coefficient: 粘性阻尼系数 (N*s/m)。
        rest_position: ``(3,)`` 平衡位置。
    """

    def __init__(
        self,
        yield_stress_min: float = 0.0,
        yield_stress_max: float = 60e3,
        piston_area: float = 1e-4,
        viscous_coefficient: float = 100.0,
        rest_position: torch.Tensor | None = None,
    ) -> None:
        self._tau_min = yield_stress_min
        self._tau_max = yield_stress_max
        self._A = piston_area
        self._c = viscous_coefficient
        self._H: float = 0.0  # 归一化磁场强度 [0, 1]
        self._rest = (
            rest_position.to(dtype=torch.float64)
            if rest_position is not None
            else torch.zeros(3, dtype=torch.float64)
        )

    def set_field_strength(self, H_normalized: float) -> None:
        """设置归一化磁场强度。

        Args:
            H_normalized: 归一化磁场强度 [0, 1]。
        """
        self._H = max(0.0, min(1.0, H_normalized))

    @property
    def yield_stress(self) -> float:
        """当前屈服应力。"""
        return self._tau_min + (self._tau_max - self._tau_min) * self._H

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算 MR 阻尼力。

        Args:
            position: ``(3,)`` 物体位置（未使用）。
            velocity: ``(3,)`` 物体速度。

        Returns:
            ``(3,)`` 阻尼力。
        """
        vel = velocity.to(dtype=torch.float64)
        v_norm = vel.norm().item()

        if v_norm < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        # Bingham 模型：F = tau_y * A * sign(v) + c * v
        tau_y = self.yield_stress
        direction = vel / v_norm
        bingham_force = tau_y * self._A * direction
        viscous_force = self._c * vel

        return -(bingham_force + viscous_force)


class FrictionPendulumRestraint(Restraint):
    """摩擦摆系统：地震隔离约束。

    模型::

        F = -W * (u/R) - mu * W * sign(v)

    其中 W 是重量，R 是曲率半径，u 是位移，mu 是摩擦系数。

    Args:
        mass: 质量 (kg)。
        curvature_radius: 摆的曲率半径 (m)。
        friction_coefficient: 滑动摩擦系数。
        gravity: 重力加速度 (m/s^2)。
    """

    def __init__(
        self,
        mass: float = 1000.0,
        curvature_radius: float = 2.0,
        friction_coefficient: float = 0.03,
        gravity: float = 9.81,
    ) -> None:
        self._mass = mass
        self._R = curvature_radius
        self._mu = friction_coefficient
        self._g = gravity

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算摩擦摆约束力。

        Args:
            position: ``(3,)`` 物体位置。
            velocity: ``(3,)`` 物体速度。

        Returns:
            ``(3,)`` 约束力。
        """
        pos = position.to(dtype=torch.float64)
        vel = velocity.to(dtype=torch.float64)

        W = self._mass * self._g

        # 恢复力（重力分量）
        restoring = -W * pos / self._R

        # 摩擦力
        v_norm = vel.norm().item()
        if v_norm > 1e-15:
            friction = -self._mu * W * vel / v_norm
        else:
            friction = torch.zeros(3, dtype=torch.float64)

        return restoring + friction

    @property
    def natural_period(self) -> float:
        """自然周期 (s)。"""
        return 2.0 * math.pi * math.sqrt(self._R / self._g)


class ParticleDamperRestraint(Restraint):
    """颗粒阻尼器：使用颗粒碰撞耗散能量。

    模型::

        F = -m_eff * e * v * (1 - exp(-t/tau))

    其中 m_eff 是等效颗粒质量，e 是恢复系数，tau 是特征时间。

    Args:
        particle_mass: 颗粒总质量 (kg)。
        coefficient_of_restitution: 恢复系数。
        characteristic_time: 特征时间常数 (s)。
        container_volume: 容器体积 (m^3)。
    """

    def __init__(
        self,
        particle_mass: float = 0.5,
        coefficient_of_restitution: float = 0.7,
        characteristic_time: float = 0.01,
        container_volume: float = 1e-4,
    ) -> None:
        self._m_p = particle_mass
        self._e = coefficient_of_restitution
        self._tau = characteristic_time
        self._V = container_volume
        self._time: float = 0.0

    def set_time(self, t: float) -> None:
        """设置当前时间。

        Args:
            t: 当前时间 (s)。
        """
        self._time = t

    @property
    def effective_mass(self) -> float:
        """等效颗粒质量（考虑填充率）。"""
        # 简化：假设 60% 填充率
        return self._m_p * 0.6

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算颗粒阻尼力。

        Args:
            position: ``(3,)`` 物体位置（未使用）。
            velocity: ``(3,)`` 物体速度。

        Returns:
            ``(3,)`` 阻尼力。
        """
        vel = velocity.to(dtype=torch.float64)
        v_norm = vel.norm().item()

        if v_norm < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        m_eff = self.effective_mass
        decay = 1.0 - math.exp(-self._time / max(self._tau, 1e-10))
        force_mag = -m_eff * self._e * v_norm * decay

        return force_mag * vel / v_norm


class NegativeStiffnessRestraint(Restraint):
    """负刚度约束：用于振动隔离的负刚度机构。

    模型::

        F = -(k_neg * x) / (1 + (x/x_max)^2)

    其中 k_neg 是负刚度，x_max 是最大位移（限制器）。
    在小位移时提供负刚度（降低系统总刚度），
    在大位移时由限制器防止失稳。

    Args:
        negative_stiffness: 负刚度值 (N/m)，应为正数。
        displacement_limit: 位移限制 (m)。
        positive_stiffness: 并联正刚度 (N/m)。
        rest_position: ``(3,)`` 平衡位置。
    """

    def __init__(
        self,
        negative_stiffness: float = 5000.0,
        displacement_limit: float = 0.01,
        positive_stiffness: float = 10000.0,
        rest_position: torch.Tensor | None = None,
    ) -> None:
        self._k_neg = negative_stiffness
        self._x_max = displacement_limit
        self._k_pos = positive_stiffness
        self._rest = (
            rest_position.to(dtype=torch.float64)
            if rest_position is not None
            else torch.zeros(3, dtype=torch.float64)
        )

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算负刚度约束力。

        Args:
            position: ``(3,)`` 物体位置。
            velocity: ``(3,)`` 物体速度（未使用）。

        Returns:
            ``(3,)`` 约束力。
        """
        pos = position.to(dtype=torch.float64)
        displacement = pos - self._rest
        x = displacement.norm().item()

        if x < 1e-30:
            return torch.zeros(3, dtype=torch.float64)

        direction = displacement / x

        # 负刚度力（带非线性限制器）
        x_ratio = x / max(self._x_max, 1e-30)
        limiter = 1.0 / (1.0 + x_ratio ** 2)
        neg_force = -self._k_neg * x * limiter

        # 正刚度力
        pos_force = -self._k_pos * x

        total_force = (neg_force + pos_force) * direction
        return total_force

    @property
    def effective_stiffness(self) -> float:
        """零位移处的有效刚度。"""
        return self._k_pos - self._k_neg

    @property
    def is_quasi_zero_stiffness(self) -> bool:
        """是否接近零刚度配置。"""
        return abs(self.effective_stiffness) < self._k_pos * 0.1
