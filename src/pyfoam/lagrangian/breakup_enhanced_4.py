"""
Enhanced breakup models v4.

Adds ETAB and SSDBreakup following OpenFOAM conventions.

- :class:`ETABBreakup`  — Enhanced Taylor Analogy Breakup model
- :class:`SSDBreakup`   — Stochastic Secondary Droplet breakup model
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.breakup import BreakupModel

__all__ = ["ETABBreakup", "SSDBreakup"]

_MIN_DIAMETER = 1e-8


class ETABBreakup(BreakupModel):
    """Enhanced Taylor Analogy Breakup (ETAB) model.

    Extends the TAB model with an improved child droplet size distribution
    and breakup regime map (Patterson & Reitz, 1998).  The deformation
    parameter y determines the breakup regime:

    - y < 1: no breakup
    - y >= 1: bag breakup, d_child = d * (1/(1 + 2*Oh))^(1/3)

    Parameters
    ----------
    k_tab : float
        Spring constant.  Default ``8.0``.
    c_tab : float
        Damping coefficient.  Default ``0.5``.
    we_crit : float
        Critical Weber number.  Default ``6.0``.
    oh_factor : float
        Ohnesorge correction factor.  Default ``2.0``.
    """

    def __init__(
        self,
        k_tab: float = 8.0,
        c_tab: float = 0.5,
        we_crit: float = 6.0,
        oh_factor: float = 2.0,
    ) -> None:
        self.k_tab = k_tab
        self.c_tab = c_tab
        self.we_crit = we_crit
        self.oh_factor = oh_factor

    def breakup(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
        particle_density: float = 1000.0,
        surface_tension: float = 0.072,
    ) -> dict:
        """Compute ETAB breakup."""
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "broken": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "broken": False}

        We = fluid_density * relative_velocity ** 2 * diameter / (2.0 * surface_tension)

        if We < self.we_crit:
            return {"diameter": diameter, "broken": False}

        Oh = fluid_viscosity / math.sqrt(
            particle_density * surface_tension * diameter
        ) if particle_density * surface_tension * diameter > 1e-30 else 0.0

        y_eq = We / 3.0
        tau = self.k_tab
        if tau < 1e-30:
            return {"diameter": diameter, "broken": False}

        y = y_eq * (1.0 - math.exp(-dt / tau))
        y = min(y, 10.0)

        if y < 1.0:
            return {"diameter": diameter, "broken": False}

        # ETAB 子液滴直径考虑 Ohnesorge 效应
        d_child = diameter / (1.0 + self.oh_factor * Oh) ** (1.0 / 3.0)
        d_child = max(d_child, _MIN_DIAMETER)

        if d_child >= diameter:
            return {"diameter": diameter, "broken": False}

        return {"diameter": d_child, "broken": True}


class SSDBreakup(BreakupModel):
    """Stochastic Secondary Droplet (SSD) breakup model.

    Uses a stochastic approach to secondary breakup where the child
    droplet diameter is drawn from a distribution based on the local
    Weber number.  This models the natural variability observed in
    experimental breakup processes.

    Parameters
    ----------
    we_min : float
        Minimum Weber number for breakup onset.  Default ``6.0``.
    d_ratio_min : float
        Minimum child/parent diameter ratio.  Default ``0.1``.
    distribution_exponent : float
        Shape parameter for child size distribution.  Default ``1.5``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        we_min: float = 6.0,
        d_ratio_min: float = 0.1,
        distribution_exponent: float = 1.5,
        seed: int | None = None,
    ) -> None:
        self.we_min = we_min
        self.d_ratio_min = d_ratio_min
        self.distribution_exponent = distribution_exponent
        self.seed = seed
        self._rng = random.Random(seed)

    def breakup(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
        particle_density: float = 1000.0,
        surface_tension: float = 0.072,
    ) -> dict:
        """Compute stochastic SSD breakup."""
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "broken": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "broken": False}

        We = fluid_density * relative_velocity ** 2 * diameter / surface_tension

        if We < self.we_min:
            return {"diameter": diameter, "broken": False}

        # 子液滴直径比基于 We 的随机采样
        # 确定性分量 + 随机分量
        d_ratio_det = self.d_ratio_min + (1.0 - self.d_ratio_min) * math.exp(-We / (self.distribution_exponent * self.we_min))
        noise = self._rng.gauss(0.0, 0.1)
        d_ratio = max(self.d_ratio_min, min(1.0, d_ratio_det + noise * d_ratio_det))

        d_child = max(diameter * d_ratio, _MIN_DIAMETER)

        if d_child >= diameter:
            return {"diameter": diameter, "broken": False}

        return {"diameter": d_child, "broken": True}
