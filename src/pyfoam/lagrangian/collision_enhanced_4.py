"""
Enhanced collision models v4.

Adds SpringDashpot and PairCollisionWall following OpenFOAM conventions.

- :class:`SpringDashpot`      — improved spring-dashpot with rolling friction
- :class:`PairCollisionWall`   — particle-wall collision with wall properties
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.collision import CollisionModel, SoftSphereModel

__all__ = ["SpringDashpot", "PairCollisionWall"]


class SpringDashpot(CollisionModel):
    """Improved spring-dashpot collision model with rolling friction.

    Extends the soft-sphere model with:
    - Rolling friction torque
    - Cohesion force (JKR adhesion model)
    - Velocity-dependent damping coefficient

    Parameters
    ----------
    spring_constant : float
        Normal spring stiffness (N/m).  Default ``1e4``.
    damping_coefficient : float
        Normal damping coefficient.  Default ``0.1``.
    tangential_friction : float
        Coulomb friction coefficient.  Default ``0.3``.
    rolling_friction : float
        Rolling friction coefficient.  Default ``0.01``.
    cohesion_energy : float
        JKR adhesion energy density (J/m²).  Default ``0.0`` (no cohesion).
    """

    def __init__(
        self,
        spring_constant: float = 1e4,
        damping_coefficient: float = 0.1,
        tangential_friction: float = 0.3,
        rolling_friction: float = 0.01,
        cohesion_energy: float = 0.0,
    ) -> None:
        if spring_constant <= 0:
            raise ValueError(f"spring_constant must be positive, got {spring_constant}")
        self.spring_constant = spring_constant
        self.damping_coefficient = damping_coefficient
        self.tangential_friction = tangential_friction
        self.rolling_friction = rolling_friction
        self.cohesion_energy = cohesion_energy

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
        """Compute collision with rolling friction and cohesion."""
        dx = [pos2[i] - pos1[i] for i in range(3)]
        dist = math.sqrt(sum(c ** 2 for c in dx))
        r1, r2 = d1 / 2.0, d2 / 2.0

        if dist >= r1 + r2 or dist < 1e-15:
            return list(vel1), list(vel2)

        overlap = (r1 + r2) - dist
        nx = [c / dist for c in dx]

        dv = [vel1[i] - vel2[i] for i in range(3)]
        v_rel_n = sum(dv[i] * nx[i] for i in range(3))

        m1 = (math.pi / 6.0) * d1 ** 3 * rho1
        m2 = (math.pi / 6.0) * d2 ** 3 * rho2
        m_total = m1 + m2
        if m_total < 1e-30:
            return list(vel1), list(vel2)
        m_eff = m1 * m2 / m_total

        # 法向力：弹簧 + 阻尼器
        sqrt_km = math.sqrt(self.spring_constant * m_eff)
        F_n = self.spring_constant * overlap

        if sqrt_km > 1e-30:
            eta_n = 2.0 * self.damping_coefficient * sqrt_km
            F_n += eta_n * v_rel_n

        # 粘附力 (JKR 简化)
        if self.cohesion_energy > 0:
            a_contact = math.sqrt(max(r1 * r2 * overlap, 0.0))
            F_adh = 3.0 * math.pi * self.cohesion_energy * a_contact
            F_n -= F_adh

        J_n = F_n * m_eff / max(self.spring_constant, 1e-30)
        dv_n = J_n / m_eff

        # 切向摩擦
        v_rel_t = [dv[i] - v_rel_n * nx[i] for i in range(3)]
        v_rel_t_mag = math.sqrt(sum(c ** 2 for c in v_rel_t))

        dv_t = 0.0
        t_hat = [0.0, 0.0, 0.0]
        if self.tangential_friction > 0 and v_rel_t_mag > 1e-15:
            J_t_max = self.tangential_friction * abs(J_n)
            J_t = min(J_t_max, m_eff * v_rel_t_mag)
            t_hat = [c / v_rel_t_mag for c in v_rel_t]
            dv_t = J_t / m_eff

        # 滚动摩擦力矩
        if self.rolling_friction > 0 and v_rel_t_mag > 1e-15:
            r_eff = r1 * r2 / max(r1 + r2, 1e-15)
            torque = self.rolling_friction * abs(F_n) * r_eff
            # 转化为等效切向速度修正
            I_eff = 0.4 * m_eff * r_eff ** 2
            if I_eff > 1e-30:
                dv_t += torque / I_eff * r_eff

        ratio1 = m2 / m_total
        ratio2 = m1 / m_total

        new_vel1 = [
            vel1[i] - ratio1 * (dv_n * nx[i] + dv_t * t_hat[i])
            for i in range(3)
        ]
        new_vel2 = [
            vel2[i] + ratio2 * (dv_n * nx[i] + dv_t * t_hat[i])
            for i in range(3)
        ]

        return new_vel1, new_vel2


class PairCollisionWall(CollisionModel):
    """Particle-wall collision model.

    Treats the second particle as a stationary wall element.  The wall
    has infinite mass and the collision uses a specified restitution and
    optional friction.

    Parameters
    ----------
    restitution : float
        Normal restitution coefficient.  Default ``0.9``.
    wall_friction : float
        Wall friction coefficient (Coulomb).  Default ``0.1``.
    """

    def __init__(
        self,
        restitution: float = 0.9,
        wall_friction: float = 0.1,
    ) -> None:
        if not 0.0 <= restitution <= 1.0:
            raise ValueError(f"restitution must be in [0, 1], got {restitution}")
        self.restitution = restitution
        self.wall_friction = wall_friction

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
        """Compute particle-wall collision.

        The wall particle (pos2) is treated as stationary (vel2 ignored).
        Only vel1 is modified.
        """
        dx = [pos2[i] - pos1[i] for i in range(3)]
        dist = math.sqrt(sum(c ** 2 for c in dx))
        r1 = d1 / 2.0
        r2 = d2 / 2.0

        if dist > r1 + r2 or dist < 1e-15:
            return list(vel1), list(vel2)

        nx = [c / dist for c in dx]
        v_n = sum(vel1[i] * nx[i] for i in range(3))

        if v_n >= 0:
            return list(vel1), list(vel2)

        # 法向反弹
        new_vel = [
            vel1[i] - (1.0 + self.restitution) * v_n * nx[i]
            for i in range(3)
        ]

        # 壁面摩擦
        if self.wall_friction > 0:
            v_t = [vel1[i] - v_n * nx[i] for i in range(3)]
            v_t_mag = math.sqrt(sum(c ** 2 for c in v_t))
            if v_t_mag > 1e-15:
                friction_impulse = self.wall_friction * abs(v_n)
                friction_impulse = min(friction_impulse, v_t_mag)
                t_hat = [c / v_t_mag for c in v_t]
                for i in range(3):
                    new_vel[i] -= friction_impulse * t_hat[i]

        return new_vel, list(vel2)
