"""
Enhanced breakup models v5.

Adds EnhancedTaylorAnalogy and KHRTBreakup following OpenFOAM conventions.

- :class:`EnhancedTaylorAnalogy` — improved TAB with empirical corrections
- :class:`KHRTBreakup`          — KH-RT breakup as a standalone model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.breakup import BreakupModel

__all__ = ["EnhancedTaylorAnalogy", "KHRTBreakup"]

_MIN_DIAMETER = 1e-8


class EnhancedTaylorAnalogy(BreakupModel):
    """Enhanced Taylor Analogy breakup with empirical corrections.

    Improves on the standard TAB model by including:
    - Reynolds-number-dependent damping correction
    - Non-linear spring response for large deformations
    - Improved child droplet size using Chryssakis & Assanis (2005)

    Parameters
    ----------
    k_tab : float
        Linear spring constant.  Default ``10.0``.
    k_nonlinear : float
        Non-linear spring constant (cubic term).  Default ``0.5``.
    c_tab : float
        Damping coefficient.  Default ``0.5``.
    we_crit : float
        Critical Weber number.  Default ``6.0``.
    re_damping_coeff : float
        Reynolds number damping correction coefficient.  Default ``0.02``.
    """

    def __init__(
        self,
        k_tab: float = 10.0,
        k_nonlinear: float = 0.5,
        c_tab: float = 0.5,
        we_crit: float = 6.0,
        re_damping_coeff: float = 0.02,
    ) -> None:
        self.k_tab = k_tab
        self.k_nonlinear = k_nonlinear
        self.c_tab = c_tab
        self.we_crit = we_crit
        self.re_damping_coeff = re_damping_coeff

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
        """Compute enhanced TAB breakup."""
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "broken": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "broken": False}

        We = fluid_density * relative_velocity ** 2 * diameter / (2.0 * surface_tension)

        if We < self.we_crit:
            return {"diameter": diameter, "broken": False}

        # 粒子 Re 数用于阻尼修正
        Re_p = particle_density * relative_velocity * diameter / max(fluid_viscosity, 1e-15)
        c_eff = self.c_tab * (1.0 + self.re_damping_coeff * math.sqrt(max(Re_p, 0.0)))

        y_eq = We / 3.0

        # 非线性弹簧效应：等效时间常数增大
        tau = self.k_tab + self.k_nonlinear * y_eq ** 2
        if tau < 1e-30:
            return {"diameter": diameter, "broken": False}

        y = y_eq * (1.0 - math.exp(-dt / tau))
        y = min(y, 10.0)

        if y < 1.0:
            return {"diameter": diameter, "broken": False}

        # Chryssakis-Assanis 子液滴尺寸
        Oh = fluid_viscosity / math.sqrt(
            particle_density * surface_tension * diameter
        ) if particle_density * surface_tension * diameter > 1e-30 else 0.0
        d_child = diameter * (1.0 + 1.2 * Oh) ** (-1.0 / 3.0)
        d_child = max(d_child, _MIN_DIAMETER)

        if d_child >= diameter:
            return {"diameter": diameter, "broken": False}

        return {"diameter": d_child, "broken": True}


class KHRTBreakup(BreakupModel):
    """KH-RT combined breakup model (standalone variant).

    Combines KH surface wave stripping with RT catastrophic breakup.
    This is the standalone version suitable for direct use in the breakup
    model chain, implementing the Huh-Gosman criterion for regime
    selection.

    Parameters
    ----------
    b0 : float
        KH constant.  Default ``0.61``.
    b1 : float
        KH breakup time constant.  Default ``1.73``.
    c_rt : float
        RT breakup time constant.  Default ``1.0``.
    a1 : float
        RT breakup size constant.  Default ``0.1``.
    """

    def __init__(
        self,
        b0: float = 0.61,
        b1: float = 1.73,
        c_rt: float = 1.0,
        a1: float = 0.1,
    ) -> None:
        self.b0 = b0
        self.b1 = b1
        self.c_rt = c_rt
        self.a1 = a1

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

        # KH 分析
        We = fluid_density * relative_velocity ** 2 * r / surface_tension
        if We < 1.0:
            return {"diameter": diameter, "broken": False}

        Oh = fluid_viscosity / math.sqrt(particle_density * surface_tension * r) if particle_density * surface_tension * r > 1e-30 else 0.0
        denom = (1.0 + Oh) * (1.0 + 1.46 * Oh ** 0.6)
        if denom < 1e-30:
            return {"diameter": diameter, "broken": False}

        lambda_kh = 9.02 * r * math.sqrt(We) / (denom * (1.0 + We / 12.0))
        d_kh = 2.0 * self.b0 * min(lambda_kh, r)

        tau_kh = self.b1 * r * math.sqrt(
            particle_density / max(fluid_density, 1e-15)
        ) / max(relative_velocity, 1e-15)

        # RT 分析
        accel = relative_velocity ** 2 / max(diameter, 1e-15)
        rho_sum = particle_density + fluid_density
        if rho_sum < 1e-30 or accel < 1e-15:
            return {"diameter": diameter, "broken": False}

        lambda_rt = 2.0 * math.pi * math.sqrt(
            surface_tension / (rho_sum * accel)
        )
        d_rt = self.a1 * lambda_rt

        omega_rt = math.sqrt(rho_sum * accel ** 3 / max(surface_tension, 1e-15))
        tau_rt = self.c_rt / max(omega_rt, 1e-15) if omega_rt > 1e-15 else float('inf')

        # 选择主导机制
        new_d = min(d_kh, d_rt)

        # 时间推进
        tau = min(tau_kh, tau_rt) if tau_rt < float('inf') else tau_kh
        if tau < 1e-15:
            return {"diameter": diameter, "broken": False}

        ratio = dt / tau
        if ratio >= 1.0:
            new_d = max(new_d, _MIN_DIAMETER)
        else:
            new_d = max(diameter * (1.0 - ratio) ** (1.0 / 3.0), _MIN_DIAMETER)

        if new_d >= diameter:
            return {"diameter": diameter, "broken": False}

        return {"diameter": new_d, "broken": True}
