"""
Constant specific internal energy thermodynamic model.

Implements a simplified thermodynamic model where Cv is constant,
as used in OpenFOAM's ``eConstThermo`` class. Enthalpy and internal
energy are linear in temperature, with the model centered on internal
energy (E) rather than enthalpy (H).

This model is the internal-energy analogue of ``hConstThermo``.

Usage::

    from pyfoam.thermophysical.econst_thermo import EConstThermo

    thermo = EConstThermo(R=287.0, Cv=718.0, Hf=0.0)
    cv = thermo.Cv(T=300.0)  # always 718.0
    e  = thermo.E(T=300.0)   # 215400.0 + Hf
"""

from __future__ import annotations

import logging

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["EConstThermo"]

logger = logging.getLogger(__name__)


class EConstThermo:
    """Constant specific internal energy thermodynamic model.

    Cv is constant (temperature-independent), making this the simplest
    internal-energy-based thermodynamic model. Internal energy and
    enthalpy are linear in T.

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg·K)). Default 287.0 (air).
    Cv : float
        Specific heat at constant volume (J/(kg·K)). Default 718.0 (air).
    Hf : float
        Heat of formation (J/kg). Default 0.0.
    Hc : float
        Heat of combustion (J/kg). Default 0.0.

    Examples::

        air = EConstThermo(R=287.0, Cv=718.0)
        assert air.Cv(300) == 718.0
        assert air.E(300) == 718.0 * 300
    """

    def __init__(
        self,
        R: float = 287.0,
        Cv: float = 718.0,
        Hf: float = 0.0,
        Hc: float = 0.0,
    ) -> None:
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if Cv <= 0:
            raise ValueError(f"Cv must be positive, got {Cv}")

        self._R = R
        self._Cv = Cv
        self._Cp = Cv + R
        self._gamma = self._Cp / Cv
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

    def Es(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Sensible specific internal energy: Es = Cv * T (without Hf).

        Args:
            T: Temperature (K).

        Returns:
            Sensible specific internal energy (J/kg).
        """
        if isinstance(T, torch.Tensor):
            return self._Cv * T
        return self._Cv * T

    def H(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Specific enthalpy: H = Cp * T + Hf = E + R*T.

        Args:
            T: Temperature (K).

        Returns:
            Specific enthalpy (J/kg).
        """
        if isinstance(T, torch.Tensor):
            return self._Cp * T + self._Hf
        return self._Cp * T + self._Hf

    def Ha(self, T: torch.Tensor | float) -> torch.Tensor | float:
        """Absolute specific enthalpy (same as H for eConstThermo).

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
            f"EConstThermo(R={self._R}, Cv={self._Cv}, "
            f"Cp={self._Cp:.1f}, gamma={self._gamma:.4f}, Hf={self._Hf})"
        )
