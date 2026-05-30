"""
Enhanced breakup models v3.

Adds TABBreakup and SHFBreakup following OpenFOAM conventions.

- :class:`TABBreakup`   — Taylor Analogy Breakup (from spray_models but standalone)
- :class:`SHFBreakup`   — Secondary Hydrodynamic Fragmentation model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.breakup import BreakupModel

__all__ = ["TABBreakup", "SHFBreakup"]

_MIN_DIAMETER = 1e-8


class TABBreakup(BreakupModel):
    """Standalone Taylor Analogy Breakup (TAB) model.

    Implements the TAB model as a breakup model (separate from spray_models)
    for use in the breakup model chain.  Droplet deformation is governed
    by a spring-mass-damper system driven by aerodynamic forces.

    Parameters
    ----------
    k_tab : float
        Spring constant (dimensionless).  Default ``10.0``.
    c_tab : float
        Damping coefficient.  Default ``0.5``.
    we_crit : float
        Critical Weber number.  Default ``6.0``.
    """

    def __init__(
        self,
        k_tab: float = 10.0,
        c_tab: float = 0.5,
        we_crit: float = 6.0,
    ) -> None:
        self.k_tab = k_tab
        self.c_tab = c_tab
        self.we_crit = we_crit

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
        """Compute TAB breakup."""
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "broken": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "broken": False}

        We = fluid_density * relative_velocity ** 2 * diameter / (2.0 * surface_tension)

        if We < self.we_crit:
            return {"diameter": diameter, "broken": False}

        y_eq = We / 3.0
        tau = self.k_tab
        if tau < 1e-30:
            return {"diameter": diameter, "broken": False}

        y = y_eq * (1.0 - math.exp(-dt / tau))
        y = min(y, 10.0)

        if y < 1.0:
            return {"diameter": diameter, "broken": False}

        factor = 6.0 * y / 5.0
        if factor <= 0.0:
            return {"diameter": diameter, "broken": False}

        d_child = max(diameter * factor ** (-1.0 / 3.0), _MIN_DIAMETER)
        if d_child >= diameter:
            return {"diameter": diameter, "broken": False}

        return {"diameter": d_child, "broken": True}


class SHFBreakup(BreakupModel):
    """Secondary Hydrodynamic Fragmentation (SHF) model.

    Models secondary breakup of droplets in high-speed crossflows using
    the instability analysis of Ranger & Nicholls (1969):

    The breakup time is:

    .. math::

        \\tau_{SHF} = C_{SHF} \\cdot d \\cdot \\sqrt{\\rho_d / \\rho_f}
                      / |U_{rel}|

    The stable child diameter is based on the critical Ohnesorge number:

    .. math::

        d_{stable} = C_d \\cdot We_{crit} \\cdot \\sigma / (\\rho_f |U_{rel}|^2)

    Parameters
    ----------
    C_SHF : float
        Breakup time constant.  Default ``2.45``.
    C_d : float
        Child droplet diameter coefficient.  Default ``0.6``.
    we_crit : float
        Critical Weber number for breakup onset.  Default ``12.0``.
    """

    def __init__(
        self,
        C_SHF: float = 2.45,
        C_d: float = 0.6,
        we_crit: float = 12.0,
    ) -> None:
        self.C_SHF = C_SHF
        self.C_d = C_d
        self.we_crit = we_crit

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
        """Compute SHF breakup."""
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "broken": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "broken": False}

        We = fluid_density * relative_velocity ** 2 * diameter / surface_tension

        if We < self.we_crit:
            return {"diameter": diameter, "broken": False}

        tau = (
            self.C_SHF
            * diameter
            / relative_velocity
            * math.sqrt(particle_density / max(fluid_density, 1e-15))
        )

        if tau < 1e-15:
            return {"diameter": diameter, "broken": False}

        d_stable = self.C_d * self.we_crit * surface_tension / (fluid_density * relative_velocity ** 2)
        d_stable = max(d_stable, _MIN_DIAMETER)

        ratio = dt / tau
        if ratio >= 1.0:
            new_d = d_stable
        else:
            new_d = max(diameter - (diameter - d_stable) * ratio, _MIN_DIAMETER)

        if new_d >= diameter:
            return {"diameter": diameter, "broken": False}

        return {"diameter": new_d, "broken": True}
