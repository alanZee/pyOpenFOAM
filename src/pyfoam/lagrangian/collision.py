"""
Binary collision models for Lagrangian particle tracking.

Models particle-particle collisions using stochastic or deterministic
approaches, providing velocity changes due to collision events.

Provides:

- :class:`CollisionModel`   — abstract base
- :class:`NoCollision`      — no collision modelling
- :class:`PairCollision`    — binary collision with restitution coefficient
- :class:`SoftSphereModel`  — soft-sphere (spring-dashpot) collision model

Usage::

    from pyfoam.lagrangian.collision import PairCollision

    model = PairCollision(restitution=0.9)
    new_v1, new_v2 = model.collide(
        pos1=[0.0, 0.0, 0.0], vel1=[1.0, 0.0, 0.0],
        d1=1e-4, rho1=1000.0,
        pos2=[1e-4, 0.0, 0.0], vel2=[-1.0, 0.0, 0.0],
        d2=1e-4, rho2=1000.0,
    )
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


__all__ = [
    "CollisionModel",
    "NoCollision",
    "PairCollision",
    "SoftSphereModel",
]


# ======================================================================
# 抽象基类
# ======================================================================

class CollisionModel(ABC):
    """Abstract base for Lagrangian collision models.

    Subclasses implement :meth:`collide`, which computes the post-collision
    velocities of two interacting particles.
    """

    @abstractmethod
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
        """Compute post-collision velocities.

        Parameters
        ----------
        pos1, pos2 : list[float]
            Positions ``[x, y, z]`` of the two particles (m).
        vel1, vel2 : list[float]
            Velocities ``[u, v, w]`` of the two particles (m/s).
        d1, d2 : float
            Diameters of the two particles (m).
        rho1, rho2 : float
            Material densities of the two particles (kg/m³).

        Returns
        -------
        tuple[list[float], list[float]]
            Post-collision velocities ``(new_vel1, new_vel2)``.
        """


# ======================================================================
# 无碰撞
# ======================================================================

class NoCollision(CollisionModel):
    """No collision modelling.

    Returns the original velocities unchanged.
    """

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
        """Return velocities unchanged."""
        return list(vel1), list(vel2)


# ======================================================================
# 二元碰撞
# ======================================================================

class PairCollision(CollisionModel):
    """Binary collision with a coefficient of restitution.

    Uses a hard-sphere collision model where the relative velocity along
    the line of centres is modified by the restitution coefficient:

    .. math::

        v'_{rel,n} = -e \\, v_{rel,n}

    The post-collision velocities are computed via conservation of momentum
    and the restitution condition:

    .. math::

        v_1' = v_1 - \\frac{m_2}{m_1 + m_2} (1 + e) \\, (v_{rel} \\cdot \\hat{n}) \\, \\hat{n}

        v_2' = v_2 + \\frac{m_1}{m_1 + m_2} (1 + e) \\, (v_{rel} \\cdot \\hat{n}) \\, \\hat{n}

    Parameters
    ----------
    restitution : float
        Coefficient of restitution ``e ∈ [0, 1]``.
        ``1`` = perfectly elastic, ``0`` = perfectly inelastic.
    """

    def __init__(self, restitution: float = 1.0) -> None:
        if not 0.0 <= restitution <= 1.0:
            raise ValueError(
                f"restitution must be in [0, 1], got {restitution}"
            )
        self.restitution = restitution

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
        """Compute post-collision velocities using hard-sphere model.

        If the particles are not in contact (centre distance > sum of
        radii), velocities are returned unchanged.
        """
        # 中心距
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        r1 = d1 / 2.0
        r2 = d2 / 2.0

        # 若粒子未接触则不处理
        if dist > r1 + r2 or dist < 1e-15:
            return list(vel1), list(vel2)

        # 连线单位法向量 (1→2)
        nx = dx / dist
        ny = dy / dist
        nz = dz / dist

        # 相对速度沿法向分量: v_rel · n
        dvx = vel1[0] - vel2[0]
        dvy = vel1[1] - vel2[1]
        dvz = vel1[2] - vel2[2]
        v_rel_n = dvx * nx + dvy * ny + dvz * nz

        # 若粒子正在分离则不处理
        if v_rel_n <= 0:
            return list(vel1), list(vel2)

        # 质量
        m1 = (math.pi / 6.0) * d1 ** 3 * rho1
        m2 = (math.pi / 6.0) * d2 ** 3 * rho2
        m_total = m1 + m2

        e = self.restitution

        # 冲量系数: (1 + e) * v_rel_n
        j = (1.0 + e) * v_rel_n

        # 更新速度
        new_vel1 = [
            vel1[0] - (m2 / m_total) * j * nx,
            vel1[1] - (m2 / m_total) * j * ny,
            vel1[2] - (m2 / m_total) * j * nz,
        ]
        new_vel2 = [
            vel2[0] + (m1 / m_total) * j * nx,
            vel2[1] + (m1 / m_total) * j * ny,
            vel2[2] + (m1 / m_total) * j * nz,
        ]

        return new_vel1, new_vel2

    def __repr__(self) -> str:
        return f"PairCollision(restitution={self.restitution})"


# ======================================================================
# 软球碰撞模型
# ======================================================================

class SoftSphereModel(CollisionModel):
    """Soft-sphere (spring-dashpot) collision model.

    Models particle-particle collisions using a linear spring-dashpot
    system.  The normal impulse is derived from the elastic potential
    energy stored in the overlap region:

    .. math::

        J_n = \\sqrt{k_n \\, m_{eff}} \\, \\delta_n

    where :math:`\\delta_n` is the overlap and :math:`m_{eff}` is the
    reduced mass.  The damping coefficient provides an effective
    restitution:

    .. math::

        e_{eff} = 1 - \\frac{\\eta_n}{2\\sqrt{k_n \\, m_{eff}}}

    clamped to ``[0, 1]``.

    An optional Coulomb friction model applies tangential sliding.

    Parameters
    ----------
    spring_constant : float
        Normal spring stiffness :math:`k_n` (N/m).  Default ``1e4``.
    damping_coefficient : float
        Normal damping coefficient :math:`\\eta_n` (N·s/m).  Default ``0.0``
        (no damping, perfectly elastic).
    tangential_friction : float
        Coulomb friction coefficient :math:`\\mu` for tangential sliding.
        Default ``0.0`` (frictionless).
    """

    def __init__(
        self,
        spring_constant: float = 1e4,
        damping_coefficient: float = 0.0,
        tangential_friction: float = 0.0,
    ) -> None:
        if spring_constant <= 0:
            raise ValueError(
                f"spring_constant must be positive, got {spring_constant}"
            )
        if damping_coefficient < 0:
            raise ValueError(
                f"damping_coefficient must be non-negative, got {damping_coefficient}"
            )
        if tangential_friction < 0:
            raise ValueError(
                f"tangential_friction must be non-negative, got {tangential_friction}"
            )
        self.spring_constant = spring_constant
        self.damping_coefficient = damping_coefficient
        self.tangential_friction = tangential_friction

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
        """Compute post-collision velocities using soft-sphere model.

        The overlap is estimated as the difference between the sum of
        radii and the centre distance.  The normal impulse is computed
        from the elastic potential energy, and an optional Coulomb
        friction impulse handles tangential sliding.

        If the particles are not in contact, velocities are returned
        unchanged.
        """
        # 中心距
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        r1 = d1 / 2.0
        r2 = d2 / 2.0

        # 若粒子未接触则不处理
        if dist >= r1 + r2 or dist < 1e-15:
            return list(vel1), list(vel2)

        # 重叠量 (穿透深度)
        overlap = (r1 + r2) - dist

        # 连线单位法向量 (1→2)
        nx = dx / dist
        ny = dy / dist
        nz = dz / dist

        # 相对速度: vel1 - vel2
        dvx = vel1[0] - vel2[0]
        dvy = vel1[1] - vel2[1]
        dvz = vel1[2] - vel2[2]

        # 法向相对速度分量 (接近时为正)
        v_rel_n = dvx * nx + dvy * ny + dvz * nz

        # 若粒子正在分离且无重叠则不处理
        if v_rel_n <= 0 and overlap < 1e-15:
            return list(vel1), list(vel2)

        # 质量
        m1 = (math.pi / 6.0) * d1 ** 3 * rho1
        m2 = (math.pi / 6.0) * d2 ** 3 * rho2
        m_total = m1 + m2
        if m_total < 1e-30:
            return list(vel1), list(vel2)
        m_eff = m1 * m2 / m_total

        # 法向冲量: J_n = sqrt(k_n * m_eff) * overlap * (1 + e_eff) / 2
        # 其中 e_eff = max(0, 1 - eta_n / (2 * sqrt(k_n * m_eff)))
        sqrt_km = math.sqrt(self.spring_constant * m_eff)
        if sqrt_km > 1e-30:
            e_eff = max(0.0, 1.0 - self.damping_coefficient / (2.0 * sqrt_km))
        else:
            e_eff = 0.0

        # 冲量大小 (沿法线 1→2 方向)
        J_n = sqrt_km * overlap * (1.0 + e_eff)

        # 法向速度修正
        dv_n = J_n / m_eff

        # 切向相对速度
        v_rel_tx = dvx - v_rel_n * nx
        v_rel_ty = dvy - v_rel_n * ny
        v_rel_tz = dvz - v_rel_n * nz
        v_rel_t_mag = math.sqrt(v_rel_tx ** 2 + v_rel_ty ** 2 + v_rel_tz ** 2)

        # 切向库仑摩擦冲量
        if self.tangential_friction > 0 and v_rel_t_mag > 1e-15:
            J_t_max = self.tangential_friction * J_n
            J_t = min(J_t_max, m_eff * v_rel_t_mag)
            t_hat_x = v_rel_tx / v_rel_t_mag
            t_hat_y = v_rel_ty / v_rel_t_mag
            t_hat_z = v_rel_tz / v_rel_t_mag
            dv_t = J_t / m_eff
        else:
            dv_t = 0.0
            t_hat_x, t_hat_y, t_hat_z = 0.0, 0.0, 0.0

        # 按质量比例分配速度修正
        ratio1 = m2 / m_total
        ratio2 = m1 / m_total

        new_vel1 = [
            vel1[0] - ratio1 * (dv_n * nx + dv_t * t_hat_x),
            vel1[1] - ratio1 * (dv_n * ny + dv_t * t_hat_y),
            vel1[2] - ratio1 * (dv_n * nz + dv_t * t_hat_z),
        ]
        new_vel2 = [
            vel2[0] + ratio2 * (dv_n * nx + dv_t * t_hat_x),
            vel2[1] + ratio2 * (dv_n * ny + dv_t * t_hat_y),
            vel2[2] + ratio2 * (dv_n * nz + dv_t * t_hat_z),
        ]

        return new_vel1, new_vel2

    def __repr__(self) -> str:
        return (
            f"SoftSphereModel(k_n={self.spring_constant}, "
            f"eta_n={self.damping_coefficient}, "
            f"mu={self.tangential_friction})"
        )
