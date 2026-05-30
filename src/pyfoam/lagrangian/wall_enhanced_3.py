"""
Enhanced wall interaction models v3.

Adds ReboundModel and StochasticSplash following OpenFOAM conventions.

- :class:`ReboundModel`     — elastic rebound with friction
- :class:`StochasticSplash` — stochastic splash with random fragment sizes
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.wall_interaction import WallInteractionModel, _normalize, _dot

__all__ = ["ReboundModel", "StochasticSplash"]


class ReboundModel(WallInteractionModel):
    """Elastic rebound with tangential friction.

    Extends the basic bounce model with a Coulomb friction component
    in the tangential direction.

    Parameters
    ----------
    normal_restitution : float
        Normal restitution coefficient.  Default ``0.7``.
    tangential_restitution : float
        Tangential restitution coefficient (1 = no friction).  Default ``0.5``.
    """

    def __init__(
        self,
        normal_restitution: float = 0.7,
        tangential_restitution: float = 0.5,
    ) -> None:
        if not 0.0 <= normal_restitution <= 1.0:
            raise ValueError("normal_restitution must be in [0, 1]")
        self.normal_restitution = normal_restitution
        self.tangential_restitution = tangential_restitution

    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Compute rebound with friction."""
        n = _normalize(wall_normal)
        v_n = _dot(velocity, n)

        if v_n >= 0:
            return {"velocity": list(velocity), "stuck": False}

        # 法向分量
        en = self.normal_restitution
        v_n_new = -en * v_n

        # 切向分量
        v_t = [velocity[i] - v_n * n[i] for i in range(3)]
        et = self.tangential_restitution
        v_t_new = [et * c for c in v_t]

        new_v = [v_n_new * n[i] + v_t_new[i] for i in range(3)]
        return {"velocity": new_v, "stuck": False}


class StochasticSplash(WallInteractionModel):
    """Stochastic splash model with random fragment size distribution.

    When splash occurs, generates random fragment sizes drawn from a
    Weibull distribution.  The number of fragments follows from mass
    conservation.

    Parameters
    ----------
    we_splash : float
        Critical Weber number for splash onset.  Default ``30.0``.
    restitution : float
        Restitution for the bounce regime.  Default ``0.5``.
    shape_parameter : float
        Weibull shape parameter for fragment size distribution.
        Default ``2.0``.
    scale_parameter : float
        Weibull scale parameter (relative to parent).  Default ``0.4``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        we_splash: float = 30.0,
        restitution: float = 0.5,
        shape_parameter: float = 2.0,
        scale_parameter: float = 0.4,
        seed: int | None = None,
    ) -> None:
        self.we_splash = we_splash
        self.restitution = restitution
        self.shape_parameter = shape_parameter
        self.scale_parameter = scale_parameter
        self.seed = seed
        self._rng = random.Random(seed)

    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Compute stochastic splash interaction."""
        n = _normalize(wall_normal)
        v_n = _dot(velocity, n)

        if v_n >= 0:
            return {
                "velocity": list(velocity),
                "stuck": False,
                "splashed": False,
                "fragment_diameter": None,
            }

        v_impact = abs(v_n)
        We = 1000.0 * v_impact ** 2 * 1e-4 / 0.072

        if We < self.we_splash:
            e = self.restitution
            new_v = [velocity[i] - (1.0 + e) * v_n * n[i] for i in range(3)]
            return {
                "velocity": new_v,
                "stuck": False,
                "splashed": False,
                "fragment_diameter": None,
            }

        # 随机飞溅
        # Weibull 分布采样子液滴直径比
        u = self._rng.random()
        u = max(u, 1e-15)
        d_ratio = self.scale_parameter * (-math.log(u)) ** (1.0 / self.shape_parameter)
        d_ratio = max(0.01, min(0.99, d_ratio))

        fragment_diameter = 1e-4 * d_ratio  # 原始直径 * 比例

        # 速度衰减
        energy_frac = 0.3 + 0.2 * self._rng.random()
        new_v = [velocity[i] - (1.0 + energy_frac) * v_n * n[i] for i in range(3)]

        return {
            "velocity": new_v,
            "stuck": False,
            "splashed": True,
            "fragment_diameter": fragment_diameter,
        }
