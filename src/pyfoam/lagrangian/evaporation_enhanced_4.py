"""
Enhanced evaporation models v4.

Adds BlowingEvaporation and NoEvaporationTwoPhase following OpenFOAM conventions.

- :class:`BlowingEvaporation`     — evaporation with blowing correction
- :class:`NoEvaporationTwoPhase`  — no evaporation with two-phase tracking
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.evaporation import EvaporationModel

__all__ = ["BlowingEvaporation", "NoEvaporationTwoPhase"]


class BlowingEvaporation(EvaporationModel):
    """Evaporation model with blowing (Stefan flow) correction.

    Accounts for the Stefan flow velocity induced by the evaporating
    vapour leaving the droplet surface.  The blowing correction factor
    modifies the heat and mass transfer coefficients:

    .. math::

        f_{blow} = \\frac{B_M}{\\ln(1 + B_M)}

    This factor reduces the transfer rates at high evaporation rates.

    Parameters
    ----------
    blowing_correction : bool
        Enable blowing correction.  Default ``True``.
    reynolds_number : float
        Particle Reynolds number.  Default ``0.0``.
    """

    def __init__(
        self,
        blowing_correction: bool = True,
        reynolds_number: float = 0.0,
    ) -> None:
        self.blowing_correction = blowing_correction
        self.reynolds_number = reynolds_number

    def evaporate(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        fluid_density: float = 1.0,
        fluid_viscosity: float = 2e-5,
        latent_heat: float = 2.26e6,
        vapour_diffusivity: float = 2.6e-5,
        thermal_conductivity: float = 0.026,
        specific_heat: float = 1005.0,
    ) -> float:
        """Compute evaporation with blowing correction."""
        if diameter < 1e-15:
            return 0.0

        dT = fluid_temperature - temperature
        if dT <= 0:
            return 0.0

        if latent_heat < 1e-15:
            return 0.0

        B_T = specific_heat * dT / latent_heat
        B_M = B_T / (1.0 + B_T) if (1.0 + B_T) > 1e-15 else 0.0

        if B_M <= 0:
            return 0.0

        Re = max(self.reynolds_number, 0.0)
        Pr = specific_heat * fluid_viscosity / max(thermal_conductivity, 1e-15)
        Sc = fluid_viscosity / max(fluid_density * vapour_diffusivity, 1e-30)

        Sh = 2.0 + 0.6 * math.sqrt(Re) * math.cbrt(Sc)

        mdot = math.pi * diameter * fluid_density * vapour_diffusivity * Sh * math.log(1.0 + B_M)

        # 吹出修正
        if self.blowing_correction and B_M > 1e-10:
            f_blow = B_M / math.log(1.0 + B_M)
            mdot /= f_blow

        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        dm = mdot * dt
        return max(min(dm, m_particle), 0.0)


class NoEvaporationTwoPhase(EvaporationModel):
    """No evaporation with two-phase tracking support.

    Returns zero mass loss but tracks the interface temperature and
    saturation state for coupling with the two-phase carrier phase model.
    This is used when evaporation is handled by the carrier phase
    Eulerian solver rather than the Lagrangian particle.

    Parameters
    ----------
    interface_temperature : float
        Interface temperature (K).  Default ``373.15``.
    saturation_ratio : float
        Vapour saturation ratio (0-1).  Default ``0.0``.
    """

    def __init__(
        self,
        interface_temperature: float = 373.15,
        saturation_ratio: float = 0.0,
    ) -> None:
        self.interface_temperature = interface_temperature
        self.saturation_ratio = saturation_ratio

    def evaporate(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        fluid_density: float = 1.0,
        fluid_viscosity: float = 2e-5,
        latent_heat: float = 2.26e6,
        vapour_diffusivity: float = 2.6e-5,
        thermal_conductivity: float = 0.026,
        specific_heat: float = 1005.0,
    ) -> float:
        """Return zero mass loss (evaporation handled by Eulerian solver)."""
        self.interface_temperature = (temperature + fluid_temperature) / 2.0
        return 0.0
