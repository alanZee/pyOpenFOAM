"""
Enhanced wall interaction models v2.

Adds BaiGosmanSplash and KuhnkeSplash following OpenFOAM conventions.

- :class:`BaiGosmanSplash` — Bai-Gosman wall splash model
- :class:`KuhnkeSplash`    — Kuhnke wall interaction model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.wall_interaction import WallInteractionModel, _normalize, _dot

__all__ = ["BaiGosmanSplash", "KuhnkeSplash"]


class BaiGosmanSplash(WallInteractionModel):
    """Bai & Gosman (1995) wall splash model.

    Uses the impact energy criterion to determine the interaction
    outcome:

    1. **Stick**: E_impact < E_stick
    2. **Rebound**: E_stick <= E_impact < E_splash
    3. **Spread**: E_impact >= E_splash (film formation)

    Parameters
    ----------
    we_stick : float
        Critical Weber number for sticking.  Default ``2.0``.
    we_spread : float
        Critical Weber number for spreading/splash.  Default ``50.0``.
    restitution : float
        Normal restitution coefficient for rebound.  Default ``0.5``.
    """

    def __init__(
        self,
        we_stick: float = 2.0,
        we_spread: float = 50.0,
        restitution: float = 0.5,
    ) -> None:
        self.we_stick = we_stick
        self.we_spread = we_spread
        self.restitution = restitution

    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Compute Bai-Gosman wall interaction."""
        n = _normalize(wall_normal)
        v_n = _dot(velocity, n)

        if v_n >= 0:
            return {"velocity": list(velocity), "stuck": False, "splashed": False}

        v_impact = abs(v_n)
        # 使用简化 We (d=1e-4, rho=1000, sigma=0.072)
        We = 1000.0 * v_impact ** 2 * 1e-4 / 0.072

        if We < self.we_stick:
            return {"velocity": [0.0, 0.0, 0.0], "stuck": True, "splashed": False}

        if We < self.we_spread:
            e = self.restitution
            new_v = [velocity[i] - (1.0 + e) * v_n * n[i] for i in range(3)]
            return {"velocity": new_v, "stuck": False, "splashed": False}

        # 飞溅
        factor = 0.3  # 能量吸收率
        new_v = [velocity[i] - (1.0 + factor) * v_n * n[i] for i in range(3)]
        return {"velocity": new_v, "stuck": False, "splashed": True}


class KuhnkeSplash(WallInteractionModel):
    """Kuhnke (1999) wall interaction model.

    Uses four regimes based on wall temperature and impact Weber number:

    1. **Dry wall deposition**: low We, low wall temperature
    2. **Rebound**: low We, high wall temperature (Leidenfrost)
    3. **Splash (thermal breakup)**: high We, high wall temperature
    4. **Splash (mechanical)**: high We, low wall temperature

    Parameters
    ----------
    we_transition : float
        Weber number regime transition.  Default ``20.0``.
    t_leidenfrost : float
        Leidenfrost temperature (K).  Default ``500.0``.
    restitution : float
        Restitution coefficient for rebound.  Default ``0.6``.
    wall_temperature : float
        Wall surface temperature (K).  Default ``300.0``.
    """

    def __init__(
        self,
        we_transition: float = 20.0,
        t_leidenfrost: float = 500.0,
        restitution: float = 0.6,
        wall_temperature: float = 300.0,
    ) -> None:
        self.we_transition = we_transition
        self.t_leidenfrost = t_leidenfrost
        self.restitution = restitution
        self.wall_temperature = wall_temperature

    def interact(
        self,
        velocity: list[float],
        wall_normal: list[float],
    ) -> dict:
        """Compute Kuhnke wall interaction."""
        n = _normalize(wall_normal)
        v_n = _dot(velocity, n)

        if v_n >= 0:
            return {"velocity": list(velocity), "stuck": False, "splashed": False}

        v_impact = abs(v_n)
        We = 1000.0 * v_impact ** 2 * 1e-4 / 0.072

        hot_wall = self.wall_temperature > self.t_leidenfrost

        # 低温壁面 + 低 We -> 粘附
        if We < self.we_transition and not hot_wall:
            return {"velocity": [0.0, 0.0, 0.0], "stuck": True, "splashed": False}

        # 高温壁面 + 低 We -> 反弹 (Leidenfrost)
        if We < self.we_transition and hot_wall:
            e = self.restitution
            new_v = [velocity[i] - (1.0 + e) * v_n * n[i] for i in range(3)]
            return {"velocity": new_v, "stuck": False, "splashed": False}

        # 高 We -> 飞溅
        energy_frac = 0.3 if hot_wall else 0.5
        new_v = [velocity[i] - (1.0 + energy_frac) * v_n * n[i] for i in range(3)]
        return {"velocity": new_v, "stuck": False, "splashed": True}
