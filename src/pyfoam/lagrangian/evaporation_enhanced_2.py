"""
Enhanced evaporation models v2.

Adds StandardEvaporation and DiffusionEvaporation following OpenFOAM conventions.

- :class:`StandardEvaporation`  — standard d²-law evaporation model
- :class:`DiffusionEvaporation` — diffusion-controlled evaporation model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.evaporation import EvaporationModel

__all__ = ["StandardEvaporation", "DiffusionEvaporation"]


class StandardEvaporation(EvaporationModel):
    """Standard d²-law evaporation model.

    Implements the classical d²-law where the droplet surface area
    decreases linearly with time:

    .. math::

        d^2(t) = d_0^2 - K \\cdot t

    The evaporation constant K depends on the thermal properties of
    the gas and liquid phases.

    Parameters
    ----------
    evaporation_constant : float
        Evaporation constant K (m²/s).  Default ``1e-6``.
    boiling_temperature : float
        Liquid boiling temperature (K).  Default ``373.15`` (water).
    """

    def __init__(
        self,
        evaporation_constant: float = 1e-6,
        boiling_temperature: float = 373.15,
    ) -> None:
        if evaporation_constant < 0:
            raise ValueError(f"evaporation_constant must be non-negative")
        self.evaporation_constant = evaporation_constant
        self.boiling_temperature = boiling_temperature

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
        """Compute mass loss via d²-law evaporation."""
        if diameter < 1e-15:
            return 0.0

        dT = fluid_temperature - temperature
        if dT <= 0:
            return 0.0

        # d²-law: d_new² = d² - K * dt
        d_new_sq = diameter ** 2 - self.evaporation_constant * dt
        if d_new_sq <= 0:
            m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
            return m_particle

        d_new = math.sqrt(d_new_sq)
        dm = (math.pi / 6.0) * (diameter ** 3 - d_new ** 3) * 1000.0
        return max(dm, 0.0)


class DiffusionEvaporation(EvaporationModel):
    """Diffusion-controlled evaporation model.

    Uses the Maxwell Stefan diffusion equation for mass transfer
    from the droplet surface:

    .. math::

        \\dot{m} = 2 \\pi d \\rho_f D_{12} Sh \\ln(1 + B_M)

    Parameters
    ----------
    schmidt_number : float
        Carrier-phase Schmidt number.  Default ``0.7``.
    molecular_weight_ratio : float
        Ratio of carrier to vapour molecular weights.  Default ``1.0``.
    """

    def __init__(
        self,
        schmidt_number: float = 0.7,
        molecular_weight_ratio: float = 1.0,
    ) -> None:
        self.schmidt_number = schmidt_number
        self.molecular_weight_ratio = molecular_weight_ratio

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
        """Compute mass loss via diffusion-controlled evaporation."""
        if diameter < 1e-15:
            return 0.0

        dT = fluid_temperature - temperature
        if dT <= 0:
            return 0.0

        # Sherwood 数 (简化: Sh = 2 for quiescent)
        Sh = 2.0

        # Spalding 质量传递数
        if latent_heat < 1e-15:
            return 0.0
        B_T = specific_heat * dT / latent_heat
        B_M = B_T / (1.0 + B_T) if (1.0 + B_T) > 1e-15 else 0.0

        if B_M <= 0:
            return 0.0

        # 质量传递速率
        mdot = 2.0 * math.pi * diameter * fluid_density * vapour_diffusivity * Sh * math.log(1.0 + B_M)

        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        dm = mdot * dt
        return max(min(dm, m_particle), 0.0)
