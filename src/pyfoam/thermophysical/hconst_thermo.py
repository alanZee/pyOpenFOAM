"""
Constant specific heat thermodynamic model.

Implements a simplified thermodynamic model where Cp is constant,
as used in OpenFOAM's ``hConstThermo`` class.

This model is suitable for ideal gases when temperature variations
are small enough that Cp can be considered constant.

Usage::

    from pyfoam.thermophysical.hconst_thermo import HConstThermo

    thermo = HConstThermo(R=287.0, Cp=1005.0, Hf=0.0)
    cp = thermo.Cp(T=300.0)  # always 1005.0
    h  = thermo.H(T=300.0)   # 301500.0 + Hf
"""

from __future__ import annotations

import logging

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["HConstThermo"]

logger = logging.getLogger(__name__)


class HConstThermo:
    """Constant specific heat thermodynamic model.

    Cp is constant (temperature-independent), making this the simplest
    thermodynamic model. Enthalpy and internal energy are linear in T.

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg·K)). Default 287.0 (air).
    Cp : float
        Specific heat at constant pressure (J/(kg·K)). Default 1005.0 (air).
    Hf : float
        Heat of formation (J/kg). Default 0.0.
    Hc : float
        Heat of combustion (J/kg). Default 0.0.

    Examples::

        air = HConstThermo(R=287.0, Cp=1005.0)
        assert air.Cp(300) == 1005.0
        assert air.H(300) == 1005.0 * 300
    """

    def __init__(
        self,
        R: float = 287.0,
        Cp: float = 1005.0,
        Hf: float = 0.0,
        Hc: float = 0.0,
    ) -> None:
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
        self._Hf = Hf
        self._Hc = Hc

    def Cp(self, T: torch.Tensor | float | None = None) -> float:
        """Specific heat at constant pressure (J/(kg·K)).

        Args:
            T: Temperature (K) — ignored (constant Cp).

        Returns:
            Constant Cp value.
        """
        return self._Cp

    def Cv(self, T: torch.Tensor | float | None = None) -> float:
        """Specific heat at constant volume (J/(kg·K)).

        Args:
            T: Temperature (K) — ignored (constant Cv).

        Returns:
            Constant Cv value.
        """
        return self._Cv

    def gamma(self, T: torch.Tensor | float | None = None) -> float:
        """Ratio of specific heats (Cp/Cv).

        Args:
            T: Temperature (K) — ignored.

        Returns:
            Constant gamma value.
        """
        return self._gamma

    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy: H = Cp * T + Hf.

        Args:
            T: Temperature (K).

        Returns:
            Specific enthalpy (J/kg).
        """
        if isinstance(T, torch.Tensor):
            return self._Cp * T + self._Hf
        return self._Cp * T + self._Hf

    def Ha(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Absolute specific enthalpy (same as H for hConstThermo).

        Args:
            T: Temperature (K).

        Returns:
            Absolute specific enthalpy (J/kg).
        """
        return self.H(T)

    def Hs(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Sensible specific enthalpy: Hs = Cp * T (without Hf).

        Args:
            T: Temperature (K).

        Returns:
            Sensible specific enthalpy (J/kg).
        """
        if isinstance(T, torch.Tensor):
            return self._Cp * T
        return self._Cp * T

    def E(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific internal energy: E = Cv * T + Hf.

        Args:
            T: Temperature (K).

        Returns:
            Specific internal energy (J/kg).
        """
        if isinstance(T, torch.Tensor):
            return self._Cv * T + self._Hf
        return self._Cv * T + self._Hf

    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""
        return self._R

    @property
    def Hf(self) -> float:
        """Heat of formation (J/kg)."""
        return self._Hf

    @property
    def Hc(self) -> float:
        """Heat of combustion (J/kg)."""
        return self._Hc

    def __repr__(self) -> str:
        return (
            f"HConstThermo(R={self._R}, Cp={self._Cp}, "
            f"Cv={self._Cv:.1f}, gamma={self._gamma:.4f}, Hf={self._Hf})"
        )
