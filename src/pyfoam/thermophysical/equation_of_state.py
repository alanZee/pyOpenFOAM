"""
Equation of State models for compressible flow.

Provides the thermodynamic relationship between pressure, density,
and temperature:

- **PerfectGas**: p = ρRT (ideal gas law)
- **IncompressiblePerfectGas**: ρ = p_ref/(RT) (density depends only on T)

These models are used by compressible solvers (rhoSimpleFoam, rhoPimpleFoam)
to close the system of equations.

Usage::

    from pyfoam.thermophysical.equation_of_state import PerfectGas

    eos = PerfectGas(R=287.0)  # air
    rho = eos.rho(p, T)        # density from pressure and temperature
    p = eos.p(rho, T)          # pressure from density and temperature
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "EquationOfState",
    "PerfectGas",
    "IncompressiblePerfectGas",
]

logger = logging.getLogger(__name__)


class EquationOfState(ABC):
    """Abstract base class for equations of state.

    Subclasses must implement :meth:`rho`, :meth:`p`, and :meth:`Cp`.
    """

    @abstractmethod
    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute density from pressure and temperature.

        Args:
            p: Pressure (Pa) — scalar or ``(n_cells,)`` tensor.
            T: Temperature (K) — scalar or ``(n_cells,)`` tensor.

        Returns:
            Density (kg/m³) — same shape as inputs.
        """

    @abstractmethod
    def p(
        self,
        rho: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute pressure from density and temperature.

        Args:
            rho: Density (kg/m³) — scalar or ``(n_cells,)`` tensor.
            T: Temperature (K) — scalar or ``(n_cells,)`` tensor.

        Returns:
            Pressure (Pa) — same shape as inputs.
        """

    @abstractmethod
    def Cp(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat capacity at constant pressure (J/(kg·K))."""

    @abstractmethod
    def Cv(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat capacity at constant volume (J/(kg·K))."""

    @abstractmethod
    def gamma(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Ratio of specific heats (Cp/Cv)."""

    @abstractmethod
    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""

    @abstractmethod
    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy h = Cp * T (J/kg)."""

    @abstractmethod
    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy e = Cv * T (J/kg)."""


class PerfectGas(EquationOfState):
    """Ideal (perfect) gas equation of state: p = ρRT.

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg·K)). Default 287.0 (air).
    Cp : float
        Specific heat at constant pressure (J/(kg·K)). Default 1005.0 (air).

    Attributes
    ----------
    R : float
        Specific gas constant.
    Cv : float
        Specific heat at constant volume = Cp - R.
    gamma : float
        Ratio of specific heats = Cp / Cv.

    Examples::

        air = PerfectGas(R=287.0, Cp=1005.0)
        rho = air.rho(p=101325.0, T=300.0)  # ~1.177 kg/m³
    """

    def __init__(self, R: float = 287.0, Cp: float = 1005.0) -> None:
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if Cp <= 0:
            raise ValueError(f"Cp must be positive, got {Cp}")
        if Cp <= R:
            raise ValueError(f"Cp must be > R, got Cp={Cp}, R={R}")

        self._R = R
        self._Cp = Cp
        self._Cv = Cp - R
        self._gamma = Cp / (Cp - R)

    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""
        return self._R

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
        return p / (self._R * T_safe)

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

        return rho * self._R * T

    def Cp(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant pressure (J/(kg·K))."""
        return self._Cp

    def Cv(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant volume (J/(kg·K))."""
        return self._Cv

    def gamma(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Ratio of specific heats (Cp/Cv)."""
        return self._gamma

    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy: h = Cp * T.

        Args:
            T: Temperature (K).

        Returns:
            Specific enthalpy (J/kg).
        """
        if isinstance(T, torch.Tensor):
            return self._Cp * T
        return self._Cp * T

    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy: e = Cv * T.

        Args:
            T: Temperature (K).

        Returns:
            Specific internal energy (J/kg).
        """
        if isinstance(T, torch.Tensor):
            return self._Cv * T
        return self._Cv * T

    def __repr__(self) -> str:
        return (
            f"PerfectGas(R={self._R}, Cp={self._Cp}, "
            f"Cv={self._Cv:.1f}, gamma={self._gamma:.4f})"
        )


class IncompressiblePerfectGas(EquationOfState):
    """Incompressible perfect gas: ρ = p_ref / (RT).

    Density depends only on temperature (not pressure), suitable for
    low-Mach-number compressible flows (natural convection, buoyancy).

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg·K)). Default 287.0 (air).
    Cp : float
        Specific heat at constant pressure (J/(kg·K)). Default 1005.0.
    p_ref : float
        Reference pressure (Pa). Default 101325.0 (1 atm).

    Examples::

        eos = IncompressiblePerfectGas(p_ref=101325.0)
        rho = eos.rho(p=101325.0, T=300.0)  # ~1.177 kg/m³
    """

    def __init__(
        self,
        R: float = 287.0,
        Cp: float = 1005.0,
        p_ref: float = 101325.0,
    ) -> None:
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if Cp <= 0:
            raise ValueError(f"Cp must be positive, got {Cp}")
        if Cp <= R:
            raise ValueError(f"Cp must be > R, got Cp={Cp}, R={R}")
        if p_ref <= 0:
            raise ValueError(f"p_ref must be positive, got {p_ref}")

        self._R = R
        self._Cp = Cp
        self._Cv = Cp - R
        self._gamma = Cp / (Cp - R)
        self._p_ref = p_ref

    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""
        return self._R

    def rho(
        self,
        p: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute density: ρ = p_ref / (RT).

        Note: pressure argument is ignored (incompressible).

        Args:
            p: Pressure (Pa) — ignored, kept for API compatibility.
            T: Temperature (K).

        Returns:
            Density (kg/m³).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1e-10)
        return self._p_ref / (self._R * T_safe)

    def p(
        self,
        rho: torch.Tensor | float,
        T: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute pressure: returns p_ref (incompressible).

        Args:
            rho: Density (kg/m³) — ignored.
            T: Temperature (K) — ignored.

        Returns:
            Reference pressure (Pa).
        """
        device = get_device()
        dtype = get_default_dtype()
        return torch.tensor(self._p_ref, dtype=dtype, device=device)

    def Cp(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant pressure (J/(kg·K))."""
        return self._Cp

    def Cv(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Specific heat at constant volume (J/(kg·K))."""
        return self._Cv

    def gamma(self, T: torch.Tensor | float | None = None) -> float | torch.Tensor:
        """Ratio of specific heats (Cp/Cv)."""
        return self._gamma

    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy: h = Cp * T."""
        if isinstance(T, torch.Tensor):
            return self._Cp * T
        return self._Cp * T

    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy: e = Cv * T."""
        if isinstance(T, torch.Tensor):
            return self._Cv * T
        return self._Cv * T

    def __repr__(self) -> str:
        return (
            f"IncompressiblePerfectGas(R={self._R}, Cp={self._Cp}, "
            f"p_ref={self._p_ref})"
        )
