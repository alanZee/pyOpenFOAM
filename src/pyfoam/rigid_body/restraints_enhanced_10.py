"""
Enhanced restraint types v10 for rigid body motion solvers.

Extends :class:`~pyfoam.rigid_body.restraints_enhanced_9` with:

- :class:`ParticleImpactDamperRestraint` -- granular particle impact damping
- :class:`ElectrorheologicalFluidRestraint` -- ER fluid with Bingham model
- :class:`NegativeStiffnessIsolator` -- quasi-zero stiffness isolator with cubic nonlinearity
- :class:`ActiveMassDamperRestraint` -- AMD with real-time optimal control

Usage::

    er = ElectrorheologicalFluidRestraint(
        yield_stress_coefficient=100.0,
    )
    er.set_field(3e3)
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
    "ParticleImpactDamperRestraint",
    "ElectrorheologicalFluidRestraint",
    "NegativeStiffnessIsolator",
    "ActiveMassDamperRestraint",
]


class ParticleImpactDamperRestraint(Restraint):
    """颗粒碰撞阻尼器：利用颗粒动量交换耗能。

    模型::

        F = -m_p * N_p * (1 + e) * v * H(|v| - v_threshold)

    Args:
        particle_mass: 单颗粒质量 (kg)。
        n_particles: 有效颗粒数。
        restitution_coefficient: 恢复系数。
        velocity_threshold: 激活速度阈值 (m/s)。
    """

    def __init__(
        self,
        particle_mass: float = 1e-4,
        n_particles: int = 100,
        restitution_coefficient: float = 0.6,
        velocity_threshold: float = 0.01,
    ) -> None:
        self._m_p = particle_mass
        self._N_p = n_particles
        self._e = restitution_coefficient
        self._v_th = velocity_threshold

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算碰撞阻尼力。"""
        vel = velocity.to(dtype=torch.float64)
        v_norm = vel.norm().item()

        if v_norm < self._v_th:
            return torch.zeros(3, dtype=torch.float64)

        direction = vel / v_norm
        F_mag = self._m_p * self._N_p * (1.0 + self._e) * v_norm
        return -F_mag * direction


class ElectrorheologicalFluidRestraint(Restraint):
    """电流变（ER）流体阻尼器：Bingham 塑性模型。

    模型::

        F = -(tau_y(E) * A * sign(v) + c * v)
        tau_y(E) = alpha * E^beta

    其中 E 是电场强度 (kV/mm)。

    Args:
        yield_stress_coefficient: 屈服应力系数 alpha。
        field_exponent: 电场指数 beta。
        piston_area: 活塞面积 (m^2)。
        viscous_coefficient: 粘性阻尼系数 (N*s/m)。
    """

    def __init__(
        self,
        yield_stress_coefficient: float = 100.0,
        field_exponent: float = 1.5,
        piston_area: float = 1e-4,
        viscous_coefficient: float = 50.0,
    ) -> None:
        self._alpha = yield_stress_coefficient
        self._beta = field_exponent
        self._A = piston_area
        self._c = viscous_coefficient
        self._E: float = 0.0  # 电场强度 (kV/mm)

    def set_field(self, E: float) -> None:
        """设置电场强度 (kV/mm)。"""
        self._E = max(0.0, E)

    @property
    def yield_stress(self) -> float:
        """当前屈服应力 (Pa)。"""
        return self._alpha * self._E ** self._beta

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算 ER 阻尼力。"""
        vel = velocity.to(dtype=torch.float64)
        v_norm = vel.norm().item()

        if v_norm < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        direction = vel / v_norm
        tau_y = self.yield_stress
        bingham = tau_y * self._A * direction
        viscous = self._c * vel

        return -(bingham + viscous)


class NegativeStiffnessIsolator(Restraint):
    """负刚度隔振器：准零刚度系统。

    模型::

        F = -k_pos * x + k_neg * x^3 / x_ref^2

    在平衡点附近有效刚度接近零。

    Args:
        positive_stiffness: 正刚度 (N/m)。
        negative_stiffness: 负刚度系数 (N/m^3)。
        reference_displacement: 参考位移 (m)。
    """

    def __init__(
        self,
        positive_stiffness: float = 1e4,
        negative_stiffness: float = 1e6,
        reference_displacement: float = 0.01,
    ) -> None:
        self._k_pos = positive_stiffness
        self._k_neg = negative_stiffness
        self._x_ref = reference_displacement

    @property
    def effective_stiffness_at_zero(self) -> float:
        """零位移处的有效刚度。"""
        return self._k_pos

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算隔振力。"""
        pos = position.to(dtype=torch.float64)
        x = pos.norm().item()

        if x < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        direction = pos / x
        F_linear = -self._k_pos * x
        F_cubic = self._k_neg * x ** 3 / max(self._x_ref ** 2, 1e-30)

        F_total = F_linear + F_cubic
        return F_total * direction


class ActiveMassDamperRestraint(Restraint):
    """主动质量阻尼器（AMD）：实时最优控制。

    模型::

        F = -k_amd * x - c_amd * v + m_amd * a_control
        a_control = -G * [x; v]  (状态反馈)

    Args:
        amd_mass: 附加质量 (kg)。
        amd_stiffness: 弹簧刚度 (N/m)。
        amd_damping: 阻尼系数 (N*s/m)。
        gain_position: 位置增益。
        gain_velocity: 速度增益。
    """

    def __init__(
        self,
        amd_mass: float = 50.0,
        amd_stiffness: float = 1e4,
        amd_damping: float = 500.0,
        gain_position: float = 1000.0,
        gain_velocity: float = 200.0,
    ) -> None:
        self._m = amd_mass
        self._k = amd_stiffness
        self._c = amd_damping
        self._Gp = gain_position
        self._Gv = gain_velocity

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """计算 AMD 力。"""
        pos = position.to(dtype=torch.float64)
        vel = velocity.to(dtype=torch.float64)

        # 被动力
        passive = -self._k * pos - self._c * vel

        # 主动力（状态反馈控制）
        a_control = -self._Gp * pos - self._Gv * vel
        active = self._m * a_control

        return passive + active
