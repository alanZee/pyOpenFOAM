"""
Enhanced collision models v2.

Adds TrajectoryModel and ORourkeCollision following OpenFOAM conventions.

- :class:`TrajectoryModel`     — trajectory-based collision detection
- :class:`ORourkeCollision`    — O'Rourke stochastic collision model
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.collision import CollisionModel, PairCollision

__all__ = ["TrajectoryModel", "ORourkeCollision"]


class TrajectoryModel(CollisionModel):
    """Trajectory-based collision model with sub-grid tracking.

    Detects collisions by computing closest approach of particle
    trajectories within the time step.  The collision criterion uses
    the minimum distance between two line segments.

    Parameters
    ----------
    restitution : float
        Coefficient of restitution.  Default ``0.9``.
    collision_efficiency : float
        Collision detection efficiency factor (0-1).  Default ``1.0``.
    """

    def __init__(
        self,
        restitution: float = 0.9,
        collision_efficiency: float = 1.0,
    ) -> None:
        if not 0.0 <= restitution <= 1.0:
            raise ValueError(f"restitution must be in [0, 1], got {restitution}")
        self._pair = PairCollision(restitution=restitution)
        self.collision_efficiency = collision_efficiency

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
        """Detect trajectory collision and compute post-collision velocities.

        Uses PairCollision for the actual momentum exchange when a
        collision is detected.
        """
        # 计算最近距离（线段近似）
        dx = [pos2[i] - pos1[i] for i in range(3)]
        dv = [vel2[i] - vel1[i] for i in range(3)]

        # 最近距离参数 t
        dv_sq = sum(c ** 2 for c in dv)
        if dv_sq < 1e-30:
            return self._pair.collide(pos1, vel1, d1, rho1, pos2, vel2, d2, rho2)

        t = -sum(dx[i] * dv[i] for i in range(3)) / dv_sq
        t = max(0.0, min(1.0, t))

        # 最近距离
        dist_sq = sum((dx[i] + t * dv[i]) ** 2 for i in range(3))
        contact_dist = (d1 + d2) / 2.0

        if dist_sq > contact_dist ** 2 / self.collision_efficiency:
            return list(vel1), list(vel2)

        # 碰撞位置
        pos1_c = [pos1[i] + t * vel1[i] for i in range(3)]
        pos2_c = [pos2[i] + t * vel2[i] for i in range(3)]

        return self._pair.collide(pos1_c, vel1, d1, rho1, pos2_c, vel2, d2, rho2)


class ORourkeCollision(CollisionModel):
    """O'Rourke stochastic collision model for spray droplets.

    Implements the stochastic collision algorithm of O'Rourke (1981)
    where collision probability depends on the relative velocity and
    cross-sectional area of particle pairs.  A uniform random number
    determines whether a collision occurs.

    Parameters
    ----------
    restitution : float
        Coefficient of restitution.  Default ``0.9``.
    collision_probability : float
        Multiplier for collision probability.  Default ``1.0``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        restitution: float = 0.9,
        collision_probability: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= restitution <= 1.0:
            raise ValueError(f"restitution must be in [0, 1], got {restitution}")
        self.restitution = restitution
        self.collision_probability = collision_probability
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
        """Stochastic collision using O'Rourke method."""
        dx = [pos2[i] - pos1[i] for i in range(3)]
        dist = math.sqrt(sum(c ** 2 for c in dx))

        r1, r2 = d1 / 2.0, d2 / 2.0

        # 碰撞截面
        cross_section = math.pi * (r1 + r2) ** 2

        # 相对速度
        dv = [vel1[i] - vel2[i] for i in range(3)]
        u_rel = math.sqrt(sum(c ** 2 for c in dv))

        if u_rel < 1e-15:
            return list(vel1), list(vel2)

        # 碰撞概率: P = cross_section * u_rel * dt (simplified, dt=1 here)
        P = self.collision_probability * cross_section * u_rel

        # 归一化到 [0,1]
        P = min(P, 1.0)

        if self._rng.random() > P:
            return list(vel1), list(vel2)

        return self._pair.collide(pos1, vel1, d1, rho1, pos2, vel2, d2, rho2)
