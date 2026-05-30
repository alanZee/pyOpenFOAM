"""
Enhanced evaporation models v3.

Adds LiquidEvaporation and MultiComponentEvaporation following OpenFOAM conventions.

- :class:`LiquidEvaporation`        — liquid fuel evaporation with saturation effects
- :class:`MultiComponentEvaporation` — multi-component fuel evaporation
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.evaporation import EvaporationModel

__all__ = ["LiquidEvaporation", "MultiComponentEvaporation"]


class LiquidEvaporation(EvaporationModel):
    """Liquid fuel evaporation with saturation vapour pressure.

    Uses the Clausius-Clapeyron relation for the saturation vapour
    pressure to compute the driving potential for evaporation:

    .. math::

        P_{sat}(T) = P_0 \\exp\\left[\\frac{L}{R/M} \\left(\\frac{1}{T_0} - \\frac{1}{T}\\right)\\right]

    Parameters
    ----------
    P0 : float
        Reference saturation pressure (Pa).  Default ``101325.0``.
    T0 : float
        Reference boiling temperature (K).  Default ``373.15``.
    molecular_weight : float
        Vapour molecular weight (kg/mol).  Default ``0.018`` (water).
    R : float
        Universal gas constant (J/(mol*K)).  Default ``8.314``.
    """

    def __init__(
        self,
        P0: float = 101325.0,
        T0: float = 373.15,
        molecular_weight: float = 0.018,
        R: float = 8.314,
    ) -> None:
        self.P0 = P0
        self.T0 = T0
        self.molecular_weight = molecular_weight
        self.R = R

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
        """Compute mass loss with saturation effects."""
        if diameter < 1e-15 or temperature < 1e-15:
            return 0.0

        dT = fluid_temperature - temperature
        if dT <= 0:
            return 0.0

        # Clausius-Clapeyron 饱和蒸汽压
        L_R = latent_heat * self.molecular_weight / max(self.R, 1e-30)
        P_sat = self.P0 * math.exp(L_R * (1.0 / self.T0 - 1.0 / temperature))

        # 驱动力
        Sh = 2.0
        B_M = P_sat / max(fluid_density * self.R / self.molecular_weight * temperature, 1e-15)

        if B_M <= 0:
            return 0.0

        mdot = math.pi * diameter * fluid_density * vapour_diffusivity * Sh * math.log(1.0 + B_M)

        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        dm = mdot * dt
        return max(min(dm, m_particle), 0.0)


class MultiComponentEvaporation(EvaporationModel):
    """Multi-component fuel evaporation model.

    Handles droplets with multiple volatile species, each with its own
    vapour pressure and mass fraction.  The total evaporation rate is
    the sum of individual species contributions.

    Parameters
    ----------
    species : list[dict]
        Species definitions, each with keys:
        - ``"Y"``: initial mass fraction (0-1)
        - ``"M"``: molecular weight (kg/mol)
        - ``"P0"``: reference saturation pressure (Pa)
        - ``"T0"``: reference boiling temperature (K)
    R : float
        Universal gas constant.  Default ``8.314``.
    """

    def __init__(
        self,
        species: list[dict] | None = None,
        R: float = 8.314,
    ) -> None:
        if species is None:
            species = [
                {"Y": 0.7, "M": 0.114, "P0": 101325.0, "T0": 371.5},  # 汽油
                {"Y": 0.3, "M": 0.018, "P0": 101325.0, "T0": 373.15},  # 水
            ]
        self.species = species
        self.R = R

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
        """Compute multi-component evaporation mass loss."""
        if diameter < 1e-15 or temperature < 1e-15:
            return 0.0

        dT = fluid_temperature - temperature
        if dT <= 0:
            return 0.0

        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        total_dm = 0.0

        for sp in self.species:
            Y = sp.get("Y", 0.0)
            M = sp.get("M", 0.018)
            P0 = sp.get("P0", 101325.0)
            T0 = sp.get("T0", 373.15)

            if Y <= 0:
                continue

            L_R = latent_heat * M / max(self.R, 1e-30)
            P_sat = P0 * math.exp(L_R * (1.0 / T0 - 1.0 / temperature))

            Sh = 2.0
            B_M = P_sat / max(fluid_density * self.R / M * temperature, 1e-15)

            if B_M <= 0:
                continue

            mdot_i = (
                math.pi * diameter * fluid_density * vapour_diffusivity
                * Sh * math.log(1.0 + B_M) * Y
            )
            total_dm += max(mdot_i * dt, 0.0)

        return max(min(total_dm, m_particle), 0.0)
