"""
Enhanced spray models v2.

Adds KHRTAtomization and LISAAtomization following OpenFOAM conventions.

- :class:`KHRTAtomization`   — KH-RT atomization model for primary breakup
- :class:`LISAAtomization`   — LISA (Linearized Instability Sheet Atomization) model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.spray_models import SprayModel

__all__ = ["KHRTAtomization", "LISAAtomization"]

_MIN_DIAMETER = 1e-10


class KHRTAtomization(SprayModel):
    """KH-RT combined atomization model for spray nozzle simulation.

    Combines KH and RT instabilities for the breakup of the intact
    liquid core near the nozzle exit.  The KH mechanism handles the
    surface wave stripping while the RT mechanism captures the
    catastrophic breakup at high acceleration.

    Parameters
    ----------
    b0 : float
        KH model constant.  Default ``0.61``.
    b1 : float
        KH breakup time constant.  Default ``1.73``.
    c_rt : float
        RT breakup constant.  Default ``1.0``.
    we_crit : float
        Critical Weber number for breakup onset.  Default ``12.0``.
    """

    def __init__(
        self,
        b0: float = 0.61,
        b1: float = 1.73,
        c_rt: float = 1.0,
        we_crit: float = 12.0,
    ) -> None:
        self.b0 = b0
        self.b1 = b1
        self.c_rt = c_rt
        self.we_crit = we_crit

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
        """Compute KH-RT atomization."""
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "atomized": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "atomized": False}

        r = diameter / 2.0
        We = fluid_density * relative_velocity ** 2 * r / surface_tension

        if We < self.we_crit:
            return {"diameter": diameter, "atomized": False}

        # KH 波长
        Oh = fluid_viscosity / math.sqrt(particle_density * surface_tension * r) if particle_density * surface_tension * r > 1e-30 else 0.0
        denom = (1.0 + Oh) * (1.0 + 1.46 * Oh ** 0.6)
        if denom < 1e-30:
            return {"diameter": diameter, "atomized": False}

        lambda_kh = 9.02 * r * math.sqrt(We) / (denom * (1.0 + We / 12.0))
        d_kh = 2.0 * self.b0 * min(lambda_kh, r)

        # RT 波长
        accel = relative_velocity ** 2 / max(diameter, 1e-15)
        rho_sum = particle_density + fluid_density
        lambda_rt = 2.0 * math.pi * math.sqrt(surface_tension / max(rho_sum * accel, 1e-30))
        d_rt = 0.1 * lambda_rt

        d_child = max(min(d_kh, d_rt), _MIN_DIAMETER)

        if d_child >= diameter:
            return {"diameter": diameter, "atomized": False}

        return {"diameter": d_child, "atomized": True}


class LISAAtomization(SprayModel):
    """LISA (Linearized Instability Sheet Atomization) model.

    Models primary atomization of liquid sheets using linear stability
    analysis of a thin liquid sheet.  The breakup length is:

    .. math::

        L_b = \\frac{d_0}{2 \\tan(\\theta)}

    and the resulting droplet diameter is related to the most
    unstable wavelength of the sheet.

    Parameters
    ----------
    sheet_angle : float
        Spray cone half-angle (degrees).  Default ``30.0``.
    C_LISA : float
        LISA model constant.  Default ``3.0``.
    we_crit : float
        Critical Weber number.  Default ``12.0``.
    """

    def __init__(
        self,
        sheet_angle: float = 30.0,
        C_LISA: float = 3.0,
        we_crit: float = 12.0,
    ) -> None:
        self.sheet_angle = sheet_angle
        self.C_LISA = C_LISA
        self.we_crit = we_crit

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
        """Compute LISA atomization."""
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "atomized": False}
        if surface_tension < 1e-15:
            return {"diameter": diameter, "atomized": False}

        We = fluid_density * relative_velocity ** 2 * diameter / (2.0 * surface_tension)
        if We < self.we_crit:
            return {"diameter": diameter, "atomized": False}

        Oh = fluid_viscosity / math.sqrt(particle_density * surface_tension * diameter) if particle_density * surface_tension * diameter > 1e-30 else 0.0

        theta_rad = math.radians(self.sheet_angle)
        tan_theta = math.tan(theta_rad)
        if tan_theta < 1e-15:
            return {"diameter": diameter, "atomized": False}

        # 液膜厚度
        h = diameter * math.sin(theta_rad)

        # 最不稳定波长 -> 子液滴直径
        d_child = self.C_LISA * math.sqrt(
            surface_tension * h / max(fluid_density * relative_velocity ** 2, 1e-30)
        )
        d_child = max(d_child, _MIN_DIAMETER)

        if d_child >= diameter:
            return {"diameter": diameter, "atomized": False}

        return {"diameter": d_child, "atomized": True}
