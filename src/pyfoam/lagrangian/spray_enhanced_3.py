"""
Enhanced spray models v3.

Adds ReitzKHRTBreakup and SSDAtomization following OpenFOAM conventions.

- :class:`ReitzKHRTBreakup` — Reitz KH-RT breakup model for secondary atomization
- :class:`SSDAtomization`   — Stochastic Sheet Droplet atomization model
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.spray_models import SprayModel

__all__ = ["ReitzKHRTBreakup", "SSDAtomization"]

_MIN_DIAMETER = 1e-10


class ReitzKHRTBreakup(SprayModel):
    """Reitz KH-RT breakup model for secondary atomization.

    Simulates secondary breakup of droplets using combined KH and
    RT instability analysis.  The breakup time is based on the
    fastest-growing wavelength of each mechanism.

    Parameters
    ----------
    b0 : float
        KH constant.  Default ``0.61``.
    b1 : float
        KH breakup time constant.  Default ``1.73``.
    c_rt : float
        RT time constant.  Default ``1.0``.
    """

    def __init__(
        self,
        b0: float = 0.61,
        b1: float = 1.73,
        c_rt: float = 1.0,
    ) -> None:
        self.b0 = b0
        self.b1 = b1
        self.c_rt = c_rt

    def atomize(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        surface_tension: float = 0.072,
        particle_density: float = 800.0,
        fluid_viscosity: float = 1.8e-5,
    ) -> dict:
        """Compute KH-RT secondary breakup."""
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "atomized": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "atomized": False}

        r = diameter / 2.0
        We = fluid_density * relative_velocity ** 2 * r / surface_tension

        if We < 1.0:
            return {"diameter": diameter, "atomized": False}

        Oh = fluid_viscosity / math.sqrt(particle_density * surface_tension * r) if particle_density * surface_tension * r > 1e-30 else 0.0

        denom = (1.0 + Oh) * (1.0 + 1.46 * Oh ** 0.6)
        if denom < 1e-30:
            return {"diameter": diameter, "atomized": False}

        lambda_kh = 9.02 * r * math.sqrt(We) / (denom * (1.0 + We / 12.0))
        tau_kh = self.b1 * r * math.sqrt(particle_density / max(fluid_density, 1e-15)) / max(relative_velocity, 1e-15)

        accel = relative_velocity ** 2 / max(diameter, 1e-15)
        rho_sum = particle_density + fluid_density
        lambda_rt = 2.0 * math.pi * math.sqrt(surface_tension / max(rho_sum * accel, 1e-30))

        omega_rt = math.sqrt(rho_sum * accel ** 3 / max(surface_tension, 1e-15))
        tau_rt = self.c_rt / max(omega_rt, 1e-15) if omega_rt > 1e-15 else float('inf')

        d_kh = 2.0 * self.b0 * min(lambda_kh, r)
        d_rt = 0.1 * lambda_rt

        tau = min(tau_kh, tau_rt) if tau_rt < float('inf') else tau_kh
        d_new = min(d_kh, d_rt)

        if tau > 1e-15:
            ratio = dt / tau
            if ratio >= 1.0:
                d_child = max(d_new, _MIN_DIAMETER)
            else:
                d_child = max(diameter * (1.0 - ratio) ** (1.0 / 3.0), _MIN_DIAMETER)
        else:
            d_child = max(d_new, _MIN_DIAMETER)

        if d_child >= diameter:
            return {"diameter": diameter, "atomized": False}

        return {"diameter": d_child, "atomized": True}


class SSDAtomization(SprayModel):
    """Stochastic Sheet Droplet (SSD) atomization model.

    Uses stochastic sampling to model the distribution of child
    droplet sizes produced by sheet breakup.  The child diameter
    distribution follows a log-normal distribution.

    Parameters
    ----------
    mean_diameter_ratio : float
        Mean child/parent diameter ratio.  Default ``0.3``.
    std_dev : float
        Standard deviation of the log-normal distribution.  Default ``0.5``.
    we_crit : float
        Critical Weber number.  Default ``5.0``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        mean_diameter_ratio: float = 0.3,
        std_dev: float = 0.5,
        we_crit: float = 5.0,
        seed: int | None = None,
    ) -> None:
        self.mean_diameter_ratio = mean_diameter_ratio
        self.std_dev = std_dev
        self.we_crit = we_crit
        self.seed = seed
        self._rng = random.Random(seed)

    def atomize(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        surface_tension: float = 0.072,
        particle_density: float = 800.0,
        fluid_viscosity: float = 1.8e-5,
    ) -> dict:
        """Compute stochastic SSD atomization."""
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "atomized": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "atomized": False}

        We = fluid_density * relative_velocity ** 2 * diameter / surface_tension
        if We < self.we_crit:
            return {"diameter": diameter, "atomized": False}

        # 对数正态采样
        log_mean = math.log(self.mean_diameter_ratio) - 0.5 * self.std_dev ** 2
        log_sample = self._rng.gauss(log_mean, self.std_dev)
        d_ratio = math.exp(log_sample)
        d_ratio = max(0.01, min(0.99, d_ratio))

        d_child = max(diameter * d_ratio, _MIN_DIAMETER)

        if d_child >= diameter:
            return {"diameter": diameter, "atomized": False}

        return {"diameter": d_child, "atomized": True}
