"""
Enhanced collision models v3.

Adds StochasticCollision and NoSeparation following OpenFOAM conventions.

- :class:`StochasticCollision` — stochastic collision with event sampling
- :class:`NoSeparation`        — collision without post-separation
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.collision import CollisionModel, PairCollision

__all__ = ["StochasticCollision", "NoSeparation"]


class StochasticCollision(CollisionModel):
    """Stochastic collision model using NTC (No Time Counter) method.

    Implements the NTC algorithm of O'Rourke & Bracco (1989) where
    the collision frequency is computed from the collision cross-section
    and relative velocity, and collision events are sampled stochastically.

    Parameters
    ----------
    restitution : float
        Coefficient of restitution.  Default ``0.9``.
    volume : float
        Cell volume for collision probability normalisation (m³).
        Default ``1e-6``.
    n_particles : int
        Number of computational parcels in the cell.  Default ``100``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        restitution: float = 0.9,
        volume: float = 1e-6,
        n_particles: int = 100,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= restitution <= 1.0:
            raise ValueError(f"restitution must be in [0, 1], got {restitution}")
        self.restitution = restitution
        self.volume = volume
        self.n_particles = n_particles
        self._pair = PairCollision(restitution=restitution)
        self.seed = seed
        self._rng = random.Random(seed)

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
        """Stochastic collision using NTC sampling."""
        r1, r2 = d1 / 2.0, d2 / 2.0
        cross_section = math.pi * (r1 + r2) ** 2

        dv = [vel1[i] - vel2[i] for i in range(3)]
        u_rel = math.sqrt(sum(c ** 2 for c in dv))

        if u_rel < 1e-15:
            return list(vel1), list(vel2)

        # NTC 碰撞频率: nu = n_p * A_cross * u_rel / V_cell
        freq = self.n_particles * cross_section * u_rel / max(self.volume, 1e-30)

        # 碰撞概率
        P = min(freq, 1.0)

        if self._rng.random() > P:
            return list(vel1), list(vel2)

        return self._pair.collide(pos1, vel1, d1, rho1, pos2, vel2, d2, rho2)


class NoSeparation(CollisionModel):
    """Collision model with no post-collision separation.

    After a collision, the particles coalesce — they share the
    mass-averaged velocity.  This models perfectly inelastic collisions
    where particles stick together (e.g., wet droplet coalescence).

    Parameters
    ----------
    coalescence_efficiency : float
        Probability of coalescence upon contact (0-1).  Default ``1.0``
        means always coalesce.
    seed : int or None
        Random seed for efficiency sampling.
    """

    def __init__(
        self,
        coalescence_efficiency: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= coalescence_efficiency <= 1.0:
            raise ValueError(f"coalescence_efficiency must be in [0, 1]")
        self.coalescence_efficiency = coalescence_efficiency
        self.seed = seed
        self._rng = random.Random(seed)

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
        """Compute perfectly inelastic collision (coalescence)."""
        dx = [pos2[i] - pos1[i] for i in range(3)]
        dist = math.sqrt(sum(c ** 2 for c in dx))
        r1, r2 = d1 / 2.0, d2 / 2.0

        if dist > r1 + r2 or dist < 1e-15:
            return list(vel1), list(vel2)

        # 检查合并效率
        if self._rng.random() > self.coalescence_efficiency:
            # 不合并，使用弹性碰撞
            e = 0.0  # 完全非弹性
            return self._perfectly_inelastic(vel1, d1, rho1, vel2, d2, rho2, e)

        # 完全合并：质量加权平均速度
        m1 = (math.pi / 6.0) * d1 ** 3 * rho1
        m2 = (math.pi / 6.0) * d2 ** 3 * rho2
        m_total = m1 + m2

        if m_total < 1e-30:
            return list(vel1), list(vel2)

        merged_vel = [
            (m1 * vel1[i] + m2 * vel2[i]) / m_total
            for i in range(3)
        ]

        return list(merged_vel), list(merged_vel)

    @staticmethod
    def _perfectly_inelastic(
        vel1, d1, rho1, vel2, d2, rho2, e
    ) -> tuple[list[float], list[float]]:
        """完全非弹性碰撞。"""
        m1 = (math.pi / 6.0) * d1 ** 3 * rho1
        m2 = (math.pi / 6.0) * d2 ** 3 * rho2
        m_total = m1 + m2

        if m_total < 1e-30:
            return list(vel1), list(vel2)

        merged = [
            (m1 * vel1[i] + m2 * vel2[i]) / m_total
            for i in range(3)
        ]
        return list(merged), list(merged)
