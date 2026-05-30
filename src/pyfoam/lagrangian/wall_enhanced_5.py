"""
Enhanced wall interaction models v5.

Adds WallBounceDistribution and CriticalVelocityModel following OpenFOAM conventions.

- :class:`WallBounceDistribution` — wall bounce with random restitution distribution
- :class:`CriticalVelocityModel` — critical velocity stick/bounce model
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.wall_interaction import WallInteractionModel, _normalize, _dot

__all__ = ["WallBounceDistribution", "CriticalVelocityModel"]


class WallBounceDistribution(WallInteractionModel):
    """Wall bounce with randomly distributed restitution coefficient.

    The normal restitution coefficient is drawn from a truncated normal
    distribution, modelling the natural variability in wall collisions.

    Parameters
    ----------
    mean_restitution : float
        Mean normal restitution coefficient.  Default ``0.7``.
    std_restitution : float
        Standard deviation of restitution distribution.  Default ``0.1``.
    min_restitution : float
        Minimum allowable restitution.  Default ``0.0``.
    max_restitution : float
        Maximum allowable restitution.  Default ``1.0``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        mean_restitution: float = 0.7,
        std_restitution: float = 0.1,
        min_restitution: float = 0.0,
        max_restitution: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.mean_restitution = mean_restitution
        self.std_restitution = std_restitution
        self.min_restitution = min_restitution
        self.max_restitution = max_restitution
        self._rng = random.Random(seed)

    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Compute bounce with stochastic restitution."""
        n = _normalize(wall_normal)
        v_n = _dot(velocity, n)

        if v_n >= 0:
            return {"velocity": list(velocity), "stuck": False}

        # 随机恢复系数
        e = self._rng.gauss(self.mean_restitution, self.std_restitution)
        e = max(self.min_restitution, min(self.max_restitution, e))

        new_v = [velocity[i] - (1.0 + e) * v_n * n[i] for i in range(3)]
        return {"velocity": new_v, "stuck": False}


class CriticalVelocityModel(WallInteractionModel):
    """Critical velocity model for stick-bounce transition.

    Particles stick if the impact velocity is below a critical velocity,
    and bounce otherwise.  The critical velocity depends on particle
    properties and surface conditions.

    Parameters
    ----------
    v_crit : float
        Critical impact velocity (m/s).  Default ``0.1``.
    restitution : float
        Normal restitution for bounce above v_crit.  Default ``0.7``.
    """

    def __init__(
        self,
        v_crit: float = 0.1,
        restitution: float = 0.7,
    ) -> None:
        self.v_crit = v_crit
        self.restitution = restitution

    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Compute critical velocity wall interaction."""
        n = _normalize(wall_normal)
        v_n = _dot(velocity, n)

        if v_n >= 0:
            return {"velocity": list(velocity), "stuck": False}

        v_impact = abs(v_n)

        if v_impact < self.v_crit:
            return {"velocity": [0.0, 0.0, 0.0], "stuck": True}

        e = self.restitution
        new_v = [velocity[i] - (1.0 + e) * v_n * n[i] for i in range(3)]
        return {"velocity": new_v, "stuck": False}
