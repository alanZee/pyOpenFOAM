"""
Enhanced breakup models v2.

Adds PilchErdman and ReitzKHRT breakup models following OpenFOAM conventions.

- :class:`PilchErdman` — Pilch-Erdman breakup time model
- :class:`ReitzKHRT`   — Reitz KH-RT combined breakup model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.breakup import BreakupModel

__all__ = ["PilchErdman", "ReitzKHRT"]

_MIN_DIAMETER = 1e-8


class PilchErdman(BreakupModel):
    """Pilch-Erdman breakup time correlation model.

    Uses the empirical correlation of Pilch & Erdman (1987) for the
    breakup time as a function of the Weber number:

    .. math::

        \\tau_b = \\frac{d}{|U_{rel}|}
                  \\cdot C_1 \\cdot We^{C_2}
                  \\cdot (1 + C_3 \\cdot Oh)^{C_4}

    where Oh is the Ohnesorge number and C1..C4 are empirical constants.

    Parameters
    ----------
    C1 : float
        Breakup time coefficient.  Default ``1.2``.
    C2 : float
        Weber number exponent.  Default ``0.25``.
    C3 : float
        Ohnesorge number coefficient.  Default ``1.0``.
    C4 : float
        Ohnesorge number exponent.  Default ``1.6``.
    stable_we : float
        Stable Weber number for post-breakup diameter.  Default ``6.0``.
    """

    def __init__(
        self,
        C1: float = 1.2,
        C2: float = 0.25,
        C3: float = 1.0,
        C4: float = 1.6,
        stable_we: float = 6.0,
    ) -> None:
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.stable_we = stable_we

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
        """Compute breakup using Pilch-Erdman time correlation."""
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "broken": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "broken": False}

        We = fluid_density * relative_velocity ** 2 * diameter / surface_tension

        if We < 6.0:
            return {"diameter": diameter, "broken": False}

        Oh = fluid_viscosity / math.sqrt(particle_density * surface_tension * diameter) if particle_density * surface_tension * diameter > 1e-30 else 0.0

        tau_b = (
            (diameter / relative_velocity)
            * self.C1 * We ** self.C2
            * (1.0 + self.C3 * Oh) ** self.C4
        )

        if tau_b < 1e-15:
            return {"diameter": diameter, "broken": False}

        ratio = dt / tau_b
        if ratio >= 1.0:
            d_stable = self.stable_we * surface_tension / (fluid_density * relative_velocity ** 2)
            new_d = max(d_stable, _MIN_DIAMETER)
        else:
            new_d = max(diameter * (1.0 - ratio) ** (1.0 / 3.0), _MIN_DIAMETER)

        if new_d >= diameter:
            return {"diameter": diameter, "broken": False}

        return {"diameter": new_d, "broken": True}


class ReitzKHRT(BreakupModel):
    """Reitz KH-RT (Kelvin-Helmholtz / Rayleigh-Taylor) combined model.

    Combines KH surface wave breakup with RT instability for dense spray
    breakup near the nozzle:

    KH breakup: dominant for bag/multimode regimes at moderate We.
    RT breakup: dominant for catastrophic breakup at high acceleration.

    The RT breakup occurs when:

    .. math::

        \\Lambda_{RT} < d

    where :math:`\\Lambda_{RT}` is the fastest-growing RT wavelength.

    Parameters
    ----------
    b0 : float
        KH model constant.  Default ``0.61``.
    b1 : float
        KH breakup time constant.  Default ``1.73``.
    c_rt : float
        RT breakup time constant.  Default ``1.0``.
    lambda_rt_max : float
        Maximum RT wavelength fraction of diameter.  Default ``0.5``.
    """

    def __init__(
        self,
        b0: float = 0.61,
        b1: float = 1.73,
        c_rt: float = 1.0,
        lambda_rt_max: float = 0.5,
    ) -> None:
        self.b0 = b0
        self.b1 = b1
        self.c_rt = c_rt
        self.lambda_rt_max = lambda_rt_max

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
        """Compute KH-RT combined breakup."""
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "broken": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "broken": False}

        r = diameter / 2.0
        We = fluid_density * relative_velocity ** 2 * r / surface_tension

        if We < 1.0:
            return {"diameter": diameter, "broken": False}

        # KH 波长和时间尺度
        Oh = fluid_viscosity / math.sqrt(particle_density * surface_tension * r) if particle_density * surface_tension * r > 1e-30 else 0.0
        We_s = math.sqrt(We)
        denom = (1.0 + Oh) * (1.0 + 1.46 * Oh ** 0.6)
        if denom < 1e-30:
            return {"diameter": diameter, "broken": False}
        lambda_kh = 9.02 * r * We_s / (denom * (1.0 + We / 12.0))
        tau_kh = self.b1 * r * math.sqrt(particle_density / fluid_density) / max(relative_velocity, 1e-15)

        # RT 波长和时间尺度
        # 加速度近似: a ~ relative_velocity^2 / d
        accel = relative_velocity ** 2 / max(diameter, 1e-15)
        density_sum = particle_density + fluid_density
        if density_sum < 1e-30 or accel < 1e-15:
            return {"diameter": diameter, "broken": False}

        lambda_rt = 2.0 * math.pi * math.sqrt(
            surface_tension / (density_sum * accel)
        )
        omega_rt = math.sqrt(density_sum * accel ** 3 / max(surface_tension, 1e-15))
        tau_rt = self.c_rt / max(omega_rt, 1e-15) if omega_rt > 1e-15 else float('inf')

        # 判断哪种机制主导
        d_kh = 2.0 * self.b0 * min(lambda_kh, r)
        d_rt = lambda_rt * self.lambda_rt_max

        # KH 时间推进
        if tau_kh > 1e-15:
            ratio_kh = dt / tau_kh
            if ratio_kh >= 1.0:
                new_d_kh = max(d_kh, _MIN_DIAMETER)
            else:
                new_d_kh = max(diameter * (1.0 - ratio_kh) ** (1.0 / 3.0), _MIN_DIAMETER)
        else:
            new_d_kh = diameter

        # RT 时间推进
        if tau_rt < float('inf') and tau_rt > 1e-15 and lambda_rt < diameter:
            ratio_rt = dt / tau_rt
            if ratio_rt >= 1.0:
                new_d_rt = max(d_rt, _MIN_DIAMETER)
            else:
                new_d_rt = max(diameter * (1.0 - ratio_rt) ** (1.0 / 3.0), _MIN_DIAMETER)
        else:
            new_d_rt = diameter

        # 取最小（最强破碎效果）
        new_d = min(new_d_kh, new_d_rt)
        new_d = max(new_d, _MIN_DIAMETER)

        if new_d >= diameter:
            return {"diameter": diameter, "broken": False}

        return {"diameter": new_d, "broken": True}
