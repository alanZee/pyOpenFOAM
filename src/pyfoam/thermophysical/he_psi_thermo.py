"""
ψ-based (psi-based) thermodynamic model for compressible solvers.

Implements the ``hePsiThermo`` model from OpenFOAM, where the
compressibility ψ = ∂ρ/∂p|_T is the primary thermodynamic variable.

For a perfect gas: ψ = 1/(RT)

This model is used in compressible solvers (rhoSimpleFoam, rhoPimpleFoam,
sonicFoam) where the pressure equation takes the form:

.. math::

    \\frac{\\partial (\\psi p)}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{U}) = 0

Usage::

    from pyfoam.thermophysical.he_psi_thermo import HePsiThermo
    from pyfoam.thermophysical.hconst_thermo import HConstThermo
    from pyfoam.thermophysical.transport_model import Sutherland

    thermo = HePsiThermo(
        thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        transport=Sutherland(),
    )
    psi = thermo.psi(p=101325.0, T=300.0)  # 1/(287*300)
    rho = thermo.rho(p=101325.0, T=300.0)  # p*psi
"""

from __future__ import annotations

import logging

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel, Sutherland

__all__ = ["HePsiThermo"]

logger = logging.getLogger(__name__)


class HePsiThermo:
    """ψ-based thermodynamic model for compressible flow.

    Combines a thermodynamic model (hConstThermo or JanafThermo) with
    a transport model and provides the compressibility ψ = 1/(RT).

    The compressibility ψ is defined as:

    .. math::

        \\psi = \\frac{\\partial \\rho}{\\partial p}\\bigg|_T
             = \\frac{1}{R T}

    And density is:

    .. math::

        \\rho = \\psi \\cdot p = \\frac{p}{R T}

    Parameters
    ----------
    thermo_model : object
        Thermodynamic model providing Cp, Cv, H, E, R methods
        (e.g. HConstThermo or JanafThermo).
    transport : TransportModel
        Transport model providing mu method (e.g. Sutherland, PolynomialTransport).
    Pr : float
        Prandtl number. Default 0.7.
    Prt : float
        Turbulent Prandtl number. Default 0.85.

    Examples::

        from pyfoam.thermophysical.hconst_thermo import HConstThermo
        from pyfoam.thermophysical.transport_model import Sutherland

        thermo = HePsiThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
            transport=Sutherland(),
        )
        psi = thermo.psi(T=300.0)         # ~1.16e-5 1/Pa
        rho = thermo.rho(p=101325, T=300) # ~1.177 kg/m³
    """

    def __init__(
        self,
        thermo_model: object | None = None,
        transport: TransportModel | None = None,
        Pr: float = 0.7,
        Prt: float = 0.85,
    ) -> None:
        # Lazy import to avoid circular imports
        from pyfoam.thermophysical.hconst_thermo import HConstThermo

        self._thermo = thermo_model or HConstThermo()
        self._transport = transport or Sutherland()
        self._Pr = Pr
        self._Prt = Prt

    # ------------------------------------------------------------------
    # Compressibility and density
    # ------------------------------------------------------------------

    def psi(self, T: torch.Tensor | float) -> torch.Tensor:
        """Compute compressibility ψ = 1/(RT).

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

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute density: ρ = ψ * p = p / (RT).

        Args:
            p: Pressure (Pa).
            T: Temperature (K).

        Returns:
            Density (kg/m³).
        """
        return self.psi(T) * p

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
            f"HePsiThermo(thermo={self._thermo!r}, "
            f"transport={self._transport!r}, Pr={self._Pr})"
        )
