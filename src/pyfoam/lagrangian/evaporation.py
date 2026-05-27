"""
Evaporation models for Lagrangian particle tracking.

Models the mass and heat transfer from a liquid droplet to the surrounding
gas phase due to evaporation.  These models reduce the particle diameter
and mass over time, coupling with the carrier-phase energy equation.

Provides:

- :class:`EvaporationModel`       — abstract base
- :class:`NoEvaporation`          — no evaporation
- :class:`RanzMarshallEvaporation` — Ranz-Marshall heat/mass transfer correlation

Usage::

    from pyfoam.lagrangian.evaporation import RanzMarshallEvaporation

    model = RanzMarshallEvaporation()
    dm = model.evaporate(
        dt=1e-4,
        diameter=1e-3,
        temperature=350.0,
        fluid_temperature=500.0,
        fluid_density=1.0,
        fluid_viscosity=2e-5,
        latent_heat=2.26e6,
        vapour_diffusivity=2.6e-5,
        thermal_conductivity=0.026,
        specific_heat=1005.0,
    )
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


__all__ = [
    "EvaporationModel",
    "NoEvaporation",
    "RanzMarshallEvaporation",
]


# ======================================================================
# 抽象基类
# ======================================================================

class EvaporationModel(ABC):
    """Abstract base for Lagrangian evaporation models.

    Subclasses implement :meth:`evaporate`, which returns the mass lost
    by a single droplet during one time step due to evaporation.
    """

    @abstractmethod
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
        """Compute mass loss due to evaporation (kg).

        Parameters
        ----------
        dt : float
            Time step (s).
        diameter : float
            Current droplet diameter (m).
        temperature : float
            Droplet surface temperature (K).
        fluid_temperature : float
            Local carrier-phase temperature (K).
        fluid_density : float
            Carrier-phase density (kg/m³).
        fluid_viscosity : float
            Carrier-phase dynamic viscosity (Pa·s).
        latent_heat : float
            Latent heat of vaporisation (J/kg).
        vapour_diffusivity : float
            Binary diffusion coefficient of vapour in the carrier phase (m²/s).
        thermal_conductivity : float
            Carrier-phase thermal conductivity (W/(m·K)).
        specific_heat : float
            Carrier-phase specific heat capacity (J/(kg·K)).

        Returns
        -------
        float
            Mass lost by the droplet during this time step (kg). Always
            non-negative; zero means no evaporation.
        """


# ======================================================================
# 无蒸发
# ======================================================================

class NoEvaporation(EvaporationModel):
    """No evaporation.

    Always returns zero mass loss.
    """

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
        """Return zero mass loss."""
        return 0.0


# ======================================================================
# Ranz-Marshall 蒸发模型
# ======================================================================

class RanzMarshallEvaporation(EvaporationModel):
    """Evaporation model using the Ranz-Marshall correlation.

    The heat and mass transfer coefficients are computed via the
    Ranz-Marshall (1952) Nusselt and Sherwood number correlations:

    .. math::

        Nu = 2.0 + 0.6 \\, Re^{1/2} \\, Pr^{1/3}

        Sh = 2.0 + 0.6 \\, Re^{1/2} \\, Sc^{1/3}

    where:

    - :math:`Re = \\rho_f |U_{rel}| d / \\mu_f` is the particle Reynolds number
    - :math:`Pr = c_p \\mu_f / k_f` is the Prandtl number
    - :math:`Sc = \\mu_f / (\\rho_f D)` is the Schmidt number

    The mass loss rate is:

    .. math::

        \\dot{m} = \\pi \\, d \\, \\rho_f \\, D \\, Sh \\, \\ln(1 + B_M)

    where :math:`B_M = (Y_{s} - Y_{\\infty}) / (1 - Y_{s})` is the
    Spalding mass transfer number, simplified here using a
    temperature-difference driving force.

    Parameters
    ----------
    reynolds_number : float
        Particle Reynolds number (default ``0.0`` for quiescent conditions).
    """

    def __init__(self, reynolds_number: float = 0.0) -> None:
        if reynolds_number < 0:
            raise ValueError(
                f"reynolds_number must be non-negative, got {reynolds_number}"
            )
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
        """Compute mass loss via Ranz-Marshall evaporation.

        Returns ``0.0`` when the droplet is hotter than the carrier phase
        (no evaporation driving force), when the diameter is negligible,
        or when the temperature difference is negligible.
        """
        if diameter < 1e-15:
            return 0.0

        dT = fluid_temperature - temperature
        if dT <= 0:
            # 载热体温度不高于液滴温度，不蒸发
            return 0.0

        Re = self.reynolds_number
        Re_sqrt = math.sqrt(max(Re, 0.0))

        # Prandtl 数
        if thermal_conductivity < 1e-15:
            return 0.0
        Pr = specific_heat * fluid_viscosity / thermal_conductivity

        # Schmidt 数
        if fluid_density * vapour_diffusivity < 1e-30:
            return 0.0
        Sc = fluid_viscosity / (fluid_density * vapour_diffusivity)

        # Ranz-Marshall: Nu = 2 + 0.6 * Re^0.5 * Pr^(1/3)
        Nu = 2.0 + 0.6 * Re_sqrt * math.cbrt(Pr)
        Sh = 2.0 + 0.6 * Re_sqrt * math.cbrt(Sc)

        # 简化 Spalding 数: B_T = cp * dT / L
        if latent_heat < 1e-15:
            return 0.0
        B_T = specific_heat * dT / latent_heat

        if B_T <= 0:
            return 0.0

        # 质量传递速率: dm/dt = -pi * d * rho_f * D * Sh * ln(1 + B_T)
        B_M = B_T / (1.0 + B_T)  # 近似 Spalding 质量传递数
        if B_M <= 0:
            return 0.0

        mdot = math.pi * diameter * fluid_density * vapour_diffusivity * Sh * math.log(1.0 + B_M)

        # 限制质量损失不超过当前液滴质量
        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0  # 假设水密度
        dm = mdot * dt
        dm = min(dm, m_particle)

        return max(dm, 0.0)

    def __repr__(self) -> str:
        return f"RanzMarshallEvaporation(Re={self.reynolds_number})"
