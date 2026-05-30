"""
Enhanced restraint types v9 for rigid body motion solvers.

Extends :class:`~pyfoam.rigid_body.restraints_enhanced_8` with:

- :class:`TunedMassDamperRestraint` -- TMD with frequency-dependent damping
- :class:`MagnetorheologicalFluidRestraint` -- MR fluid damper with Bingham-plastic model
- :class:`FrictionPendulumIsolator` -- triple friction pendulum seismic isolator
- :class:`ActiveTendonRestraint` -- active tendon with real-time force control

Usage::

    tmd = TunedMassDamperRestraint(
        primary_frequency=2.0,
        mass_ratio=0.05,
    )
    force = tmd.force(position, velocity)

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` restraint models
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.restraints import Restraint

__all__ = [
    "TunedMassDamperRestraint",
    "MagnetorheologicalFluidRestraint",
    "FrictionPendulumIsolator",
    "ActiveTendonRestraint",
]


class TunedMassDamperRestraint(Restraint):
    """调谐质量阻尼器（TMD）：频率依赖的附加质量阻尼系统。

    模型::

        F = -k_TMD * (x - x_d) - c_TMD * (v - v_d)
        k_TMD = m_d * (2 * pi * f_opt)^2
        c_TMD = 2 * zeta_opt * m_d * (2 * pi * f_opt)

    Args:
        primary_frequency: 主结构固有频率 (Hz)。
        mass_ratio: 附加质量与主结构质量比。
        damping_ratio: 优化阻尼比。
        total_mass: 主结构总质量 (kg)。
    """

    def __init__(
        self,
        primary_frequency: float = 2.0,
        mass_ratio: float = 0.05,
        damping_ratio: float = 0.15,
        total_mass: float = 1000.0,
    ) -> None:
        self._f_p = primary_frequency
        self._mu = mass_ratio
        self._zeta = damping_ratio
        self._M = total_mass

        # 优化 TMD 频率
        f_opt = self._f_p / (1.0 + self._mu)
        omega_opt = 2.0 * math.pi * f_opt

        self._m_d = self._mu * self._M
        self._k_TMD = self._m_d * omega_opt ** 2
        self._c_TMD = 2.0 * self._zeta * self._m_d * omega_opt

        self._displacement: torch.Tensor = torch.zeros(3, dtype=torch.float64)
        self._velocity_d: torch.Tensor = torch.zeros(3, dtype=torch.float64)

    def set_state(
        self,
        displacement: torch.Tensor,
        velocity: torch.Tensor,
    ) -> None:
        """设置 TMD 状态。"""
        self._displacement = displacement.to(dtype=torch.float64)
        self._velocity_d = velocity.to(dtype=torch.float64)

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算 TMD 力。"""
        pos = position.to(dtype=torch.float64)
        vel = velocity.to(dtype=torch.float64)

        spring_force = -self._k_TMD * (pos - self._displacement)
        damping_force = -self._c_TMD * (vel - self._velocity_d)

        return spring_force + damping_force

    @property
    def tmd_frequency(self) -> float:
        """TMD 频率 (Hz)。"""
        return self._f_p / (1.0 + self._mu)

    @property
    def tmd_mass(self) -> float:
        """TMD 质量 (kg)。"""
        return self._m_d


class MagnetorheologicalFluidRestraint(Restraint):
    """磁流变（MR）流体阻尼器：Bingham 塑性模型。

    模型::

        F = -(tau_y(B) * A * sign(v) + c * v^n)
        tau_y(B) = tau_0 + alpha * B^beta

    Args:
        yield_stress_base: 零磁场屈服应力 (Pa)。
        yield_stress_coefficient: 磁场屈服应力系数。
        field_exponent: 磁场指数。
        piston_area: 活塞面积 (m^2)。
        viscous_coefficient: 粘性阻尼系数。
        velocity_exponent: 速度指数。
    """

    def __init__(
        self,
        yield_stress_base: float = 0.0,
        yield_stress_coefficient: float = 50.0,
        field_exponent: float = 1.5,
        piston_area: float = 1e-4,
        viscous_coefficient: float = 100.0,
        velocity_exponent: float = 1.0,
    ) -> None:
        self._tau_0 = yield_stress_base
        self._alpha = yield_stress_coefficient
        self._beta = field_exponent
        self._A = piston_area
        self._c = viscous_coefficient
        self._n = velocity_exponent
        self._B: float = 0.0  # 磁感应强度 (T)

    def set_field(self, B: float) -> None:
        """设置磁感应强度 (T)。"""
        self._B = max(0.0, B)

    @property
    def yield_stress(self) -> float:
        """当前屈服应力。"""
        return self._tau_0 + self._alpha * self._B ** self._beta

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算 MR 阻尼力。"""
        vel = velocity.to(dtype=torch.float64)
        v_norm = vel.norm().item()

        if v_norm < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        direction = vel / v_norm
        tau_y = self.yield_stress

        bingham = tau_y * self._A * direction
        viscous = self._c * (v_norm ** self._n) * direction

        return -(bingham + viscous)


class FrictionPendulumIsolator(Restraint):
    """三重摩擦摆隔震器：多曲面摩擦摆模型。

    模型::

        F = mu * N * sign(v) + K_eff * x
        K_eff = W / R_eff

    其中 W 是重量，R_eff 是等效曲率半径。

    Args:
        friction_coefficient: 摩擦系数。
        effective_radius: 等效曲率半径 (m)。
        weight: 载荷重量 (N)。
        initial_stiffness: 初始刚度 (N/m)。
    """

    def __init__(
        self,
        friction_coefficient: float = 0.05,
        effective_radius: float = 2.0,
        weight: float = 10000.0,
        initial_stiffness: float = 1e8,
    ) -> None:
        self._mu = friction_coefficient
        self._R = effective_radius
        self._W = weight
        self._K_init = initial_stiffness

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算隔震力。"""
        pos = position.to(dtype=torch.float64)
        vel = velocity.to(dtype=torch.float64)
        x = pos.norm().item()
        v = vel.norm().item()

        if x < 1e-15 and v < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        # 摩擦力
        if v > 1e-15:
            friction_dir = vel / v
            friction_force = self._mu * self._W * friction_dir
        else:
            friction_force = torch.zeros(3, dtype=torch.float64)

        # 恢复力（重力偏心）
        if x > 1e-15:
            K_eff = self._W / max(self._R, 1e-10)
            restore_dir = pos / x
            restore_force = K_eff * x * restore_dir
        else:
            restore_force = torch.zeros(3, dtype=torch.float64)

        return -friction_force - restore_force

    @property
    def effective_period(self) -> float:
        """等效周期 (s)。"""
        return 2.0 * math.pi * math.sqrt(self._R / 9.81)


class ActiveTendonRestraint(Restraint):
    """主动腱约束：实时力控制的主动拉索。

    模型::

        F = -k_tendon * x - c_tendon * v + F_control
        F_control = K_p * e + K_i * integral(e) + K_d * de/dt

    Args:
        tendon_stiffness: 腱刚度 (N/m)。
        tendon_damping: 腱阻尼 (N*s/m)。
        K_p: PID 比例增益。
        K_i: PID 积分增益。
        K_d: PID 微分增益。
    """

    def __init__(
        self,
        tendon_stiffness: float = 1e5,
        tendon_damping: float = 500.0,
        K_p: float = 1000.0,
        K_i: float = 100.0,
        K_d: float = 50.0,
    ) -> None:
        self._k = tendon_stiffness
        self._c = tendon_damping
        self._Kp = K_p
        self._Ki = K_i
        self._Kd = K_d
        self._integral_error: torch.Tensor = torch.zeros(3, dtype=torch.float64)
        self._prev_error: torch.Tensor | None = None
        self._dt: float = 0.001
        self._target: torch.Tensor = torch.zeros(3, dtype=torch.float64)

    def set_target(self, target: torch.Tensor) -> None:
        """设置控制目标。"""
        self._target = target.to(dtype=torch.float64)

    def set_dt(self, dt: float) -> None:
        self._dt = max(dt, 1e-10)

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算主动腱力。"""
        pos = position.to(dtype=torch.float64)
        vel = velocity.to(dtype=torch.float64)

        # 被动力
        passive = -self._k * pos - self._c * vel

        # PID 控制
        error = self._target - pos
        self._integral_error = self._integral_error + error * self._dt

        derivative = torch.zeros(3, dtype=torch.float64)
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / self._dt
        self._prev_error = error.clone()

        control = self._Kp * error + self._Ki * self._integral_error + self._Kd * derivative

        return passive + control

    def reset_state(self) -> None:
        """重置 PID 状态。"""
        self._integral_error.zero_()
        self._prev_error = None
