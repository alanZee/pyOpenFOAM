"""
Thermophysical package combining equation of state and transport models.

Provides a unified interface for thermodynamic properties used by
compressible solvers:

- Density from EOS: ρ = ρ(p, T)
- Viscosity from transport: μ = μ(T)
- Thermal conductivity: κ = μ * Cp / Pr
- Enthalpy and internal energy

Usage::

    from pyfoam.thermophysical.thermo import BasicThermo

    thermo = BasicThermo()  # air: PerfectGas + Sutherland
    rho = thermo.rho(p, T)
    mu = thermo.mu(T)
    kappa = thermo.kappa(T)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import (
    EquationOfState,
    PerfectGas,
    IncompressiblePerfectGas,
)
from pyfoam.thermophysical.transport_model import (
    TransportModel,
    ConstantViscosity,
    Sutherland,
)

__all__ = [
    "BasicThermo",
    "create_thermo",
    "create_air_thermo",
]

logger = logging.getLogger(__name__)


class BasicThermo:
    """Combined thermophysical model (EOS + transport).

    Wraps an equation of state and a transport model into a single
    interface used by compressible solvers.

    Parameters
    ----------
    eos : EquationOfState
        Equation of state (e.g. PerfectGas, IncompressiblePerfectGas).
    transport : TransportModel
        Transport model (e.g. Sutherland, ConstantViscosity).
    Pr : float
        Prandtl number (dimensionless). Default 0.7 (air).
    Prt : float
        Turbulent Prandtl number. Default 0.85.

    Examples::

        thermo = BasicThermo(PerfectGas(), Sutherland())
        rho = thermo.rho(p, T)
        mu = thermo.mu(T)
    """

    def __init__(
        self,
        eos: EquationOfState | None = None,
        transport: TransportModel | None = None,
        Pr: float = 0.7,
        Prt: float = 0.85,
    ) -> None:
        self._eos = eos or PerfectGas()
        self._transport = transport or Sutherland()
        self._Pr = Pr
        self._Prt = Prt

    # ------------------------------------------------------------------
    # EOS delegates
    # ------------------------------------------------------------------

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute density from pressure and temperature."""
        return self._eos.rho(p, T)

    def p(
        self,
        rho: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute pressure from density and temperature."""
        return self._eos.p(rho, T)

    def Cp(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant pressure."""
        return self._eos.Cp(T)

    def Cv(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant volume."""
        return self._eos.Cv(T)

    def gamma(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Ratio of specific heats."""
        return self._eos.gamma(T)

    def R(self) -> float:
        """Specific gas constant."""
        return self._eos.R()

    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy."""
        return self._eos.H(T)

    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy."""
        return self._eos.E(T)

    # ------------------------------------------------------------------
    # Transport delegates
    # ------------------------------------------------------------------

    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Dynamic viscosity from transport model."""
        return self._transport.mu(T)

    def nu(
        self,
        T: torch.Tensor | float,
        rho: torch.Tensor | float,
    ) -> torch.Tensor:
        """Kinematic viscosity: ν = μ / ρ."""
        return self._transport.nu(T, rho)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def kappa(self, T: torch.Tensor | float) -> torch.Tensor:
        """Thermal conductivity: κ = μ * Cp / Pr.

        Args:
            T: Temperature (K).

        Returns:
            Thermal conductivity (W/(m·K)).
        """
        mu = self._transport.mu(T)
        cp = self._eos.Cp(T)
        if isinstance(cp, torch.Tensor):
            return mu * cp / self._Pr
        return mu * (cp / self._Pr)

    def alpha(
        self,
        T: torch.Tensor | float,
        rho: torch.Tensor | float,
    ) -> torch.Tensor:
        """Thermal diffusivity: α = κ / (ρ * Cp).

        Args:
            T: Temperature (K).
            rho: Density (kg/m³).

        Returns:
            Thermal diffusivity (m²/s).
        """
        kappa = self.kappa(T)
        cp = self._eos.Cp(T)
        if isinstance(cp, torch.Tensor):
            return kappa / (rho * cp)
        return kappa / (rho * cp)

    @property
    def Pr(self) -> float:
        """Prandtl number."""
        return self._Pr

    @property
    def Prt(self) -> float:
        """Turbulent Prandtl number."""
        return self._Prt

    @property
    def eos(self) -> EquationOfState:
        """The equation of state."""
        return self._eos

    @property
    def transport(self) -> TransportModel:
        """The transport model."""
        return self._transport

    def __repr__(self) -> str:
        return (
            f"BasicThermo(eos={self._eos!r}, "
            f"transport={self._transport!r}, "
            f"Pr={self._Pr})"
        )


def create_thermo(
    eos_type: str = "perfectGas",
    transport_type: str = "sutherland",
    **kwargs: Any,
) -> BasicThermo:
    """Factory function to create a BasicThermo from string names.

    Args:
        eos_type: EOS type — ``"perfectGas"`` or ``"incompressiblePerfectGas"``.
        transport_type: Transport type — ``"constant"`` or ``"sutherland"``.
        **kwargs: Passed to EOS and transport constructors.

    Returns:
        A configured :class:`BasicThermo` instance.
    """
    # EOS
    eos_map: dict[str, type[EquationOfState]] = {
        "perfectGas": PerfectGas,
        "incompressiblePerfectGas": IncompressiblePerfectGas,
    }
    if eos_type not in eos_map:
        raise ValueError(
            f"Unknown EOS type '{eos_type}'. "
            f"Available: {list(eos_map.keys())}"
        )

    # Extract EOS kwargs
    eos_kwarg_keys = {"R", "Cp", "p_ref"}
    eos_kwargs = {k: v for k, v in kwargs.items() if k in eos_kwarg_keys}
    eos = eos_map[eos_type](**eos_kwargs)

    # Transport
    transport_map: dict[str, type[TransportModel]] = {
        "constant": ConstantViscosity,
        "sutherland": Sutherland,
    }
    if transport_type not in transport_map:
        raise ValueError(
            f"Unknown transport type '{transport_type}'. "
            f"Available: {list(transport_map.keys())}"
        )

    transport_kwarg_keys = {"mu", "mu_ref", "T_ref", "S"}
    transport_kwargs = {k: v for k, v in kwargs.items() if k in transport_kwarg_keys}
    transport = transport_map[transport_type](**transport_kwargs)

    # Pr
    Pr = kwargs.get("Pr", 0.7)
    Prt = kwargs.get("Prt", 0.85)

    return BasicThermo(eos=eos, transport=transport, Pr=Pr, Prt=Prt)


def create_air_thermo(
    R: float = 287.0,
    Cp: float = 1005.0,
    mu_ref: float = 1.716e-5,
    T_ref: float = 273.15,
    S: float = 110.4,
    Pr: float = 0.7,
) -> BasicThermo:
    """Create a thermophysical model for air using standard defaults.

    Combines PerfectGas EOS with Sutherland transport.

    Args:
        R: Specific gas constant (J/(kg·K)).
        Cp: Specific heat at constant pressure (J/(kg·K)).
        mu_ref: Reference viscosity (Pa·s).
        T_ref: Reference temperature (K).
        S: Sutherland constant (K).
        Pr: Prandtl number.

    Returns:
        A :class:`BasicThermo` configured for air.
    """
    return BasicThermo(
        eos=PerfectGas(R=R, Cp=Cp),
        transport=Sutherland(mu_ref=mu_ref, T_ref=T_ref, S=S),
        Pr=Pr,
    )
