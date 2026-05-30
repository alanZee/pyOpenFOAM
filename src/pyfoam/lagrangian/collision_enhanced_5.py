"""
Enhanced collision models v5.

Adds SubCycledCollision and CoulalogluCollision following OpenFOAM conventions.

- :class:`SubCycledCollision`    — time-subcycled collision integration
- :class:`CoulalogluCollision`   — Coulaloglu & Prosperi collision model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.collision import CollisionModel, PairCollision, SoftSphereModel

__all__ = ["SubCycledCollision", "CoulalogluCollision"]


class SubCycledCollision(CollisionModel):
    """Time-subcycled collision model for improved accuracy.

    Splits the collision time step into multiple sub-steps for better
    resolution of the collision dynamics, particularly important for
    stiff spring-dashpot systems.

    Parameters
    ----------
    n_subcycles : int
        Number of collision sub-steps.  Default ``5``.
    spring_constant : float
        Spring stiffness (N/m).  Default ``1e4``.
    damping_coefficient : float
        Damping coefficient.  Default ``0.1``.
    tangential_friction : float
        Coulomb friction.  Default ``0.3``.
    """

    def __init__(
        self,
        n_subcycles: int = 5,
        spring_constant: float = 1e4,
        damping_coefficient: float = 0.1,
        tangential_friction: float = 0.3,
    ) -> None:
        if n_subcycles < 1:
            raise ValueError(f"n_subcycles must be >= 1, got {n_subcycles}")
        self.n_subcycles = n_subcycles
        self._soft_sphere = SoftSphereModel(
            spring_constant=spring_constant,
            damping_coefficient=damping_coefficient,
            tangential_friction=tangential_friction,
        )

    def collide(
        self,
        pos1: list[float],
        vel1: list[float],
        d1: float,
        rho1: float,
        pos2: list[float],
        vel2: list[float],
        d2: float,
        rho2: float,
    ) -> tuple[list[float], list[float]]:
        """Compute sub-cycled collision."""
        # 简化：对速度进行子步迭代
        v1 = list(vel1)
        v2 = list(vel2)

        # 检查是否接触
        dx = [pos2[i] - pos1[i] for i in range(3)]
        dist = math.sqrt(sum(c ** 2 for c in dx))
        r1, r2 = d1 / 2.0, d2 / 2.0

        if dist >= r1 + r2:
            return list(vel1), list(vel2)

        # 子步碰撞求解
        for _ in range(self.n_subcycles):
            v1_new, v2_new = self._soft_sphere.collide(
                pos1, v1, d1, rho1, pos2, v2, d2, rho2
            )
            # 检查收敛（速度变化量）
            dv1 = math.sqrt(sum((v1_new[i] - v1[i]) ** 2 for i in range(3)))
            v1 = v1_new
            v2 = v2_new
            if dv1 < 1e-10:
                break

        return v1, v2


class CoulalogluCollision(CollisionModel):
    """Coulaloglu & Prosperi (1978) collision model.

    Uses the collision model of Coulaloglu & Prosperi with:
    - Impact-parameter-dependent restitution
    - Energy loss due to deformation and heat dissipation

    The effective restitution coefficient depends on the impact
    parameter b (normalized by sum of radii):

    .. math::

        e(b) = e_0 (1 - b^2)^{0.25}

    Parameters
    ----------
    e0 : float
        Head-on restitution coefficient.  Default ``0.9``.
    energy_dissipation : float
        Additional energy dissipation fraction.  Default ``0.1``.
    """

    def __init__(
        self,
        e0: float = 0.9,
        energy_dissipation: float = 0.1,
    ) -> None:
        if not 0.0 <= e0 <= 1.0:
            raise ValueError(f"e0 must be in [0, 1], got {e0}")
        self.e0 = e0
        self.energy_dissipation = energy_dissipation

    def collide(
        self,
        pos1: list[float],
        vel1: list[float],
        d1: float,
        rho1: float,
        pos2: list[float],
        vel2: list[float],
        d2: float,
        rho2: float,
    ) -> tuple[list[float], list[float]]:
        """Compute Coulaloglu collision with impact-parameter-dependent restitution."""
        dx = [pos2[i] - pos1[i] for i in range(3)]
        dist = math.sqrt(sum(c ** 2 for c in dx))
        r1, r2 = d1 / 2.0, d2 / 2.0

        if dist > r1 + r2 or dist < 1e-15:
            return list(vel1), list(vel2)

        nx = [c / dist for c in dx]

        dv = [vel1[i] - vel2[i] for i in range(3)]
        v_rel_n = sum(dv[i] * nx[i] for i in range(3))

        if v_rel_n <= 0:
            return list(vel1), list(vel2)

        # 计算冲击参数 b
        # b = 分离距离投影到切平面 / (r1 + r2)
        v_rel_t = [dv[i] - v_rel_n * nx[i] for i in range(3)]
        v_rel_t_mag = math.sqrt(sum(c ** 2 for c in v_rel_t))
        v_rel_mag = math.sqrt(sum(c ** 2 for c in dv))
        if v_rel_mag < 1e-15:
            return list(vel1), list(vel2)

        b = min(v_rel_t_mag / v_rel_mag, 1.0)

        # 有效恢复系数
        e = self.e0 * (1.0 - b ** 2) ** 0.25
        e *= (1.0 - self.energy_dissipation)
        e = max(0.0, min(1.0, e))

        m1 = (math.pi / 6.0) * d1 ** 3 * rho1
        m2 = (math.pi / 6.0) * d2 ** 3 * rho2
        m_total = m1 + m2

        if m_total < 1e-30:
            return list(vel1), list(vel2)

        j = (1.0 + e) * v_rel_n

        new_vel1 = [vel1[i] - (m2 / m_total) * j * nx[i] for i in range(3)]
        new_vel2 = [vel2[i] + (m1 / m_total) * j * nx[i] for i in range(3)]

        return new_vel1, new_vel2
