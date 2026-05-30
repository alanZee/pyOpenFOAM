"""
Enhanced evaporation models v5.

Adds EvaporationDI and FrosslingEvaporation following OpenFOAM conventions.

- :class:`EvaporationDI`      — direct injection evaporation model
- :class:`FrosslingEvaporation` — Frossling correlation evaporation model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.evaporation import EvaporationModel

__all__ = ["EvaporationDI", "FrosslingEvaporation"]


class EvaporationDI(EvaporationModel):
    """Direct injection evaporation model.

    Combines the Ranz-Marshall correlation with a fuel-specific
    vapour pressure curve for diesel fuel injection applications.

    Parameters
    ----------
    fuel_vapour_pressure : float
        Fuel vapour pressure at reference conditions (Pa).  Default ``5000.0``.
    reference_temperature : float
        Reference temperature for vapour pressure (K).  Default ``350.0``.
    evaporation_exponent : float
        Temperature exponent for vapour pressure.  Default ``0.5``.
    """

    def __init__(
        self,
        fuel_vapour_pressure: float = 5000.0,
        reference_temperature: float = 350.0,
        evaporation_exponent: float = 0.5,
    ) -> None:
        self.fuel_vapour_pressure = fuel_vapour_pressure
        self.reference_temperature = reference_temperature
        self.evaporation_exponent = evaporation_exponent

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
        """Compute diesel fuel evaporation."""
        if diameter < 1e-15:
            return 0.0

        dT = fluid_temperature - temperature
        if dT <= 0:
            return 0.0

        T_ref = self.reference_temperature
        if T_ref < 1e-15:
            return 0.0

        # 温度相关的蒸汽压
        P_vap = self.fuel_vapour_pressure * (temperature / T_ref) ** self.evaporation_exponent

        # 简化驱动力
        if fluid_density < 1e-15 or temperature < 1e-15:
            return 0.0

        B_M = P_vap / (fluid_density * 287.0 * temperature)
        if B_M <= 0:
            return 0.0

        Sh = 2.0
        mdot = math.pi * diameter * fluid_density * vapour_diffusivity * Sh * math.log(1.0 + B_M)

        m_particle = (math.pi / 6.0) * diameter ** 3 * 800.0  # 柴油密度
        dm = mdot * dt
        return max(min(dm, m_particle), 0.0)


class FrosslingEvaporation(EvaporationModel):
    """Frossling correlation evaporation model.

    Uses the Frossling (1938) correlation for the Sherwood number:

    .. math::

        Sh = 2 + 0.552 Re^{1/2} Sc^{1/3}

    This is commonly used for evaporating droplets in convective flows
    and is more accurate than Ranz-Marshall at high Re.

    Parameters
    ----------
    reynolds_number : float
        Particle Reynolds number.  Default ``0.0``.
    """

    def __init__(
        self,
        reynolds_number: float = 0.0,
    ) -> None:
        if reynolds_number < 0:
            raise ValueError(f"reynolds_number must be non-negative")
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
        """Compute evaporation using Frossling correlation."""
        if diameter < 1e-15:
            return 0.0

        dT = fluid_temperature - temperature
        if dT <= 0:
            return 0.0

        Re = max(self.reynolds_number, 0.0)

        # Schmidt 数
        if fluid_density * vapour_diffusivity < 1e-30:
            return 0.0
        Sc = fluid_viscosity / (fluid_density * vapour_diffusivity)

        # Frossling: Sh = 2 + 0.552 * Re^0.5 * Sc^(1/3)
        Sh = 2.0 + 0.552 * math.sqrt(Re) * math.cbrt(Sc)

        if latent_heat < 1e-15:
            return 0.0
        B_T = specific_heat * dT / latent_heat
        B_M = B_T / (1.0 + B_T) if (1.0 + B_T) > 1e-15 else 0.0

        if B_M <= 0:
            return 0.0

        mdot = math.pi * diameter * fluid_density * vapour_diffusivity * Sh * math.log(1.0 + B_M)

        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        dm = mdot * dt
        return max(min(dm, m_particle), 0.0)
