"""
ρ-based (rho-based) thermodynamic model for compressible solvers.

Implements the ``heRhoThermo`` model from OpenFOAM, where density ρ
is the primary thermodynamic variable computed from the equation of state.

This model is used in compressible solvers where density is stored
directly and pressure is derived:

.. math::

    \\rho = \\frac{p}{R T}

    p = \\rho R T

Usage::

    from pyfoam.thermophysical.he_rho_thermo import HeRhoThermo
    from pyfoam.thermophysical.hconst_thermo import HConstThermo
    from pyfoam.thermophysical.transport_model import Sutherland

    thermo = HeRhoThermo(
        thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        transport=Sutherland(),
    )
    rho = thermo.rho(p=101325.0, T=300.0)  # p/(RT) ≈ 1.177
    p = thermo.p(rho=1.2, T=300.0)          # ρRT ≈ 103320
"""

from __future__ import annotations

import logging

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel, Sutherland

__all__ = ["HeRhoThermo"]

logger = logging.getLogger(__name__)


class HeRhoThermo:
    """ρ-based thermodynamic model for compressible flow.

    Combines a thermodynamic model with a transport model and provides
    density from the ideal gas law: ρ = p/(RT).

    This is the "rho" variant used by rhoSimpleFoam and rhoPimpleFoam,
    where density is the stored variable and pressure is computed.

    Parameters
    ----------
    thermo_model : object
        Thermodynamic model providing Cp, Cv, H, E, R methods
        (e.g. HConstThermo or JanafThermo).
    transport : TransportModel
        Transport model providing mu method.
    Pr : float
        Prandtl number. Default 0.7.
    Prt : float
        Turbulent Prandtl number. Default 0.85.

    Examples::

        from pyfoam.thermophysical.hconst_thermo import HConstThermo
        from pyfoam.thermophysical.transport_model import Sutherland

        thermo = HeRhoThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
            transport=Sutherland(),
        )
        rho = thermo.rho(p=101325.0, T=300.0)
        p = thermo.p(rho=1.2, T=300.0)
    """

    def __init__(
        self,
        thermo_model: object | None = None,
        transport: TransportModel | None = None,
        Pr: float = 0.7,
        Prt: float = 0.85,
    ) -> None:
        from pyfoam.thermophysical.hconst_thermo import HConstThermo

        self._thermo = thermo_model or HConstThermo()
        self._transport = transport or Sutherland()
        self._Pr = Pr
        self._Prt = Prt

    # ------------------------------------------------------------------
    # Density and pressure
    # ------------------------------------------------------------------

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute density: ρ = p / (RT).

        Args:
            p: Pressure (Pa).
            T: Temperature (K).

        Returns:
            Density (kg/m³).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=dtype, device=device)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1e-10)
        R = self._thermo.R()
        return p / (R * T_safe)

    def p(
        self,
        rho: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute pressure: p = ρRT.

        Args:
            rho: Density (kg/m³).
            T: Temperature (K).

        Returns:
            Pressure (Pa).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(rho, torch.Tensor):
            rho = torch.tensor(rho, dtype=dtype, device=device)
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        R = self._thermo.R()
        return rho * R * T

    def psi(self, T: torch.Tensor | float) -> torch.Tensor:
        """Compute compressibility ψ = 1/(RT).

        For perfect gas: ψ = ∂ρ/∂p|_T = 1/(RT)

        Args:
            T: Temperature (K).

        Returns:
            Compressibility ψ (1/Pa).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1e-10)
        R = self._thermo.R()
        return 1.0 / (R * T_safe)

    # ------------------------------------------------------------------
    # Thermodynamic delegates
    # ------------------------------------------------------------------

    def Cp(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant pressure."""
        return self._thermo.Cp(T)

    def Cv(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant volume."""
        return self._thermo.Cv(T)

    def gamma(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Ratio of specific heats."""
        return self._thermo.gamma(T)

    def R(self) -> float:
        """Specific gas constant."""
        return self._thermo.R()

    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy."""
        return self._thermo.H(T)

    def Hs(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Sensible specific enthalpy."""
        if hasattr(self._thermo, 'Hs'):
            return self._thermo.Hs(T)
        return self._thermo.H(T)

    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy."""
        return self._thermo.E(T)

    # ------------------------------------------------------------------
    # Transport delegates
    # ------------------------------------------------------------------

    def mu(self, T: torch.Tensor | float) -> torch.Tensor:
        """Dynamic viscosity."""
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
        cp = self._thermo.Cp(T)
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
        cp = self._thermo.Cp(T)
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
    def thermo(self) -> object:
        """The underlying thermodynamic model."""
        return self._thermo

    @property
    def transport(self) -> TransportModel:
        """The transport model."""
        return self._transport

    def __repr__(self) -> str:
        return (
            f"HeRhoThermo(thermo={self._thermo!r}, "
            f"transport={self._transport!r}, Pr={self._Pr})"
        )
