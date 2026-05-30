"""
Enhanced restraint types v8 for rigid body motion solvers.

Extends :class:`~pyfoam.rigid_body.restraints_enhanced_7` with:

- :class:`PneumaticHybridRestraint` -- pneumatic-hybrid damper with gas spring coupling
- :class:`ElectrorheologicalRestraint` -- ER fluid damper with field-dependent viscosity
- :class:`InerterRestraint` -- mechanical inerter (force proportional to relative acceleration)
- :class:`QuasiZeroStiffnessRestraint` -- QZS isolator with positive and negative stiffness in parallel

Usage::

    er = ElectrorheologicalRestraint(
        yield_stress_min=0.0,
        yield_stress_max=30e3,
    )
    er.set_field_strength(0.6)
    force = er.force(position, velocity)

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` restraint models
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.restraints import Restraint

__all__ = [
    "PneumaticHybridRestraint",
    "ElectrorheologicalRestraint",
    "InerterRestraint",
    "QuasiZeroStiffnessRestraint",
]


class PneumaticHybridRestraint(Restraint):
    """气动混合阻尼器：气弹簧与液压阻尼的并联系统。

    模型::

        F = -k_gas * x^n_gas - c_hydraulic * v - F_preload

    其中 k_gas 是气弹簧刚度，n_gas 是多变指数，
    c_hydraulic 是液压阻尼系数。

    Args:
        gas_stiffness: 气弹簧刚度 (N/m)。
        polytropic_index: 多变指数 (1.0 = 等温, 1.4 = 绝热)。
        hydraulic_damping: 液压阻尼系数 (N*s/m)。
        preload_force: 预载力 (N)。
        stroke_limit: 行程限制 (m)。
    """

    def __init__(
        self,
        gas_stiffness: float = 5000.0,
        polytropic_index: float = 1.4,
        hydraulic_damping: float = 500.0,
        preload_force: float = 100.0,
        stroke_limit: float = 0.1,
    ) -> None:
        self._k_gas = gas_stiffness
        self._n_gas = polytropic_index
        self._c_hyd = hydraulic_damping
        self._F_pre = preload_force
        self._x_max = stroke_limit

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算气动混合阻尼力。

        Args:
            position: ``(3,)`` 物体位置。
            velocity: ``(3,)`` 物体速度。

        Returns:
            ``(3,)`` 阻尼力。
        """
        pos = position.to(dtype=torch.float64)
        vel = velocity.to(dtype=torch.float64)
        x = pos.norm().item()

        if x < 1e-30 and vel.norm().item() < 1e-30:
            return torch.zeros(3, dtype=torch.float64)

        direction = pos / max(x, 1e-30)

        # 气弹簧力（非线性）
        x_clamped = min(x, self._x_max)
        gas_force = self._k_gas * x_clamped ** self._n_gas

        # 液压阻尼力
        damping_force = self._c_hyd * vel.norm().item()

        total_mag = gas_force + self._F_pre
        if vel.norm().item() > 1e-15:
            vel_dir = vel / vel.norm()
            total = -total_mag * direction - damping_force * vel_dir
        else:
            total = -total_mag * direction

        return total

    @property
    def effective_stroke(self) -> float:
        """有效行程限制。"""
        return self._x_max


class ElectrorheologicalRestraint(Restraint):
    """电流变（ER）流体阻尼器：电场相关的粘度阻尼器。

    模型::

        F = -(tau_y(E) * A * sign(v) + c * v)

    其中 tau_y(E) 是与电场强度相关的屈服应力，
    A 是活塞面积，c 是粘性阻尼系数。

    与 MR 阻尼器类似，但使用电场而非磁场。

    Args:
        yield_stress_min: 零电场屈服应力 (Pa)。
        yield_stress_max: 饱和电场屈服应力 (Pa)。
        piston_area: 活塞面积 (m^2)。
        viscous_coefficient: 粘性阻尼系数 (N*s/m)。
        rest_position: ``(3,)`` 平衡位置。
    """

    def __init__(
        self,
        yield_stress_min: float = 0.0,
        yield_stress_max: float = 30e3,
        piston_area: float = 1e-4,
        viscous_coefficient: float = 50.0,
        rest_position: torch.Tensor | None = None,
    ) -> None:
        self._tau_min = yield_stress_min
        self._tau_max = yield_stress_max
        self._A = piston_area
        self._c = viscous_coefficient
        self._E_normalized: float = 0.0  # 归一化电场 [0, 1]
        self._rest = (
            rest_position.to(dtype=torch.float64)
            if rest_position is not None
            else torch.zeros(3, dtype=torch.float64)
        )

    def set_field_strength(self, E_normalized: float) -> None:
        """设置归一化电场强度。

        Args:
            E_normalized: 归一化电场强度 [0, 1]。
        """
        self._E_normalized = max(0.0, min(1.0, E_normalized))

    @property
    def yield_stress(self) -> float:
        """当前屈服应力。"""
        return self._tau_min + (self._tau_max - self._tau_min) * self._E_normalized

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算 ER 阻尼力。

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

        tau_y = self.yield_stress
        direction = vel / v_norm
        bingham_force = tau_y * self._A * direction
        viscous_force = self._c * vel

        return -(bingham_force + viscous_force)


class InerterRestraint(Restraint):
    """机械惯容器：力与相对加速度成正比。

    模型::

        F = b * a_relative

    其中 b 是惯容系数 (kg)，a_relative 是相对加速度。

    与传统质量不同，惯容器的两端都可以运动。

    Args:
        inertance: 惯容系数 (kg)。
        damping_coefficient: 附加阻尼 (N*s/m)。
        rest_position: ``(3,)`` 平衡位置。
    """

    def __init__(
        self,
        inertance: float = 100.0,
        damping_coefficient: float = 10.0,
        rest_position: torch.Tensor | None = None,
    ) -> None:
        self._b = inertance
        self._c = damping_coefficient
        self._rest = (
            rest_position.to(dtype=torch.float64)
            if rest_position is not None
            else torch.zeros(3, dtype=torch.float64)
        )
        self._prev_velocity: torch.Tensor | None = None
        self._dt: float = 0.001

    def set_dt(self, dt: float) -> None:
        """设置时间步长（用于数值加速度估计）。

        Args:
            dt: 时间步长 (s)。
        """
        self._dt = max(dt, 1e-10)

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算惯容器力。

        Args:
            position: ``(3,)`` 物体位置（未使用）。
            velocity: ``(3,)`` 物体速度。

        Returns:
            ``(3,)`` 惯容器力。
        """
        vel = velocity.to(dtype=torch.float64)

        # 数值加速度估计
        acceleration = torch.zeros(3, dtype=torch.float64)
        if self._prev_velocity is not None:
            acceleration = (vel - self._prev_velocity) / self._dt

        self._prev_velocity = vel.clone()

        # F = b * a + c * v
        return self._b * acceleration + self._c * vel

    @property
    def inertance(self) -> float:
        """惯容系数。"""
        return self._b


class QuasiZeroStiffnessRestraint(Restraint):
    """准零刚度（QZS）约束：正负刚度并联实现极低动刚度。

    模型::

        F = -k_pos * x + k_neg * x / (1 + (x/x_s)^2) - c * v

    在平衡点附近实现准零刚度（动刚度接近零），
    同时保持大位移时的正刚度（承载能力）。

    Args:
        positive_stiffness: 正弹簧刚度 (N/m)。
        negative_stiffness: 负刚度值 (N/m)。
        stabiliser_displacement: 稳定化位移 (m)。
        damping_coefficient: 阻尼系数 (N*s/m)。
        rest_position: ``(3,)`` 平衡位置。
    """

    def __init__(
        self,
        positive_stiffness: float = 15000.0,
        negative_stiffness: float = 14000.0,
        stabiliser_displacement: float = 0.01,
        damping_coefficient: float = 200.0,
        rest_position: torch.Tensor | None = None,
    ) -> None:
        self._k_pos = positive_stiffness
        self._k_neg = negative_stiffness
        self._x_s = stabiliser_displacement
        self._c = damping_coefficient
        self._rest = (
            rest_position.to(dtype=torch.float64)
            if rest_position is not None
            else torch.zeros(3, dtype=torch.float64)
        )

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算准零刚度约束力。

        Args:
            position: ``(3,)`` 物体位置。
            velocity: ``(3,)`` 物体速度。

        Returns:
            ``(3,)`` 约束力。
        """
        pos = position.to(dtype=torch.float64)
        vel = velocity.to(dtype=torch.float64)
        displacement = pos - self._rest
        x = displacement.norm().item()

        if x < 1e-30 and vel.norm().item() < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        if x < 1e-30:
            return -self._c * vel

        direction = displacement / x

        # 正刚度力
        pos_force = -self._k_pos * x

        # 负刚度力（带稳定器）
        x_ratio = x / max(self._x_s, 1e-30)
        neg_force = self._k_neg * x / (1.0 + x_ratio ** 2)

        # 阻尼力
        damping = -self._c * vel

        total = (pos_force + neg_force) * direction + damping
        return total

    @property
    def effective_static_stiffness(self) -> float:
        """平衡点处的有效静刚度。"""
        return self._k_pos - self._k_neg

    @property
    def dynamic_stiffness_at_origin(self) -> float:
        """原点处的动刚度（应接近零）。"""
        # dF/dx 在 x=0 处
        return self._k_pos - self._k_neg

    @property
    def is_quasi_zero(self) -> bool:
        """是否为准零刚度配置。"""
        return abs(self.effective_static_stiffness) < self._k_pos * 0.1
