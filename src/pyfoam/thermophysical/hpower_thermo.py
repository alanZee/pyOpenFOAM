"""
Power-law enthalpy thermodynamic model.

Implements a thermodynamic model where Cp follows a power-law
temperature dependence, as used in OpenFOAM's ``hPowerThermo`` class.

.. math::

    C_p(T) = C_{p,0} \\cdot T^n

This model is useful for approximating temperature-dependent specific
heat in simplified combustion or reacting flow simulations.

Usage::

    from pyfoam.thermophysical.hpower_thermo import HPowerThermo

    thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.1)
    cp = thermo.Cp(T=300.0)  # 1005.0 * 300^0.1
    h  = thermo.H(T=300.0)
"""

from __future__ import annotations

import logging

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["HPowerThermo"]

logger = logging.getLogger(__name__)


class HPowerThermo:
    """Power-law enthalpy thermodynamic model.

    Specific heat follows a power law in temperature:

    .. math::

        C_p(T) = C_{p,0} \\cdot T^n

    where n is the exponent (0 gives constant Cp).

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg·K)). Default 287.0 (air).
    Cp0 : float
        Reference specific heat coefficient (J/(kg·K)). Default 1005.0.
    exponent : float
        Power-law exponent (dimensionless). Default 0.0 (constant Cp).
    Hf : float
        Heat of formation (J/kg). Default 0.0.
    Hc : float
        Heat of combustion (J/kg). Default 0.0.

    Examples::

        # Constant Cp (exponent=0)
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.0)
        assert thermo.Cp(300) == 1005.0

        # Power-law Cp
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.1)
        cp = thermo.Cp(T=300.0)  # 1005.0 * 300^0.1
    """

    def __init__(
        self,
        R: float = 287.0,
        Cp0: float = 1005.0,
        exponent: float = 0.0,
        Hf: float = 0.0,
        Hc: float = 0.0,
    ) -> None:
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if Cp0 <= 0:
            raise ValueError(f"Cp0 must be positive, got {Cp0}")

        self._R = R
        self._Cp0 = Cp0
        self._n = exponent
        self._Hf = Hf
        self._Hc = Hc

    def Cp(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific heat at constant pressure: Cp = Cp0 * T^n.

        Args:
            T: Temperature (K).

        Returns:
            Specific heat capacity (J/(kg·K)).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1.0)
        return self._Cp0 * T_safe.pow(self._n)

    def Cv(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific heat at constant volume: Cv = Cp - R.

        Args:
            T: Temperature (K).

        Returns:
            Specific heat at constant volume (J/(kg·K)).
        """
        return self.Cp(T) - self._R

    def gamma(self, T: torch.Tensor | float) -> torch.Tensor:
        """Ratio of specific heats: gamma = Cp / Cv.

        Args:
            T: Temperature (K).

        Returns:
            Ratio of specific heats (dimensionless).
        """
        cp = self.Cp(T)
        cv = cp - self._R
        return cp / cv

    def H(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific enthalpy: H = Cp0/(n+1) * T^(n+1) + Hf.

        For n != -1. For n = -1 (singular), returns Cp0 * ln(T) + Hf.

        Args:
            T: Temperature (K).

        Returns:
            Specific enthalpy (J/kg).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1.0)

        if abs(self._n + 1.0) < 1e-12:
            # Special case: n = -1
            return self._Cp0 * torch.log(T_safe) + self._Hf

        return self._Cp0 / (self._n + 1.0) * T_safe.pow(self._n + 1.0) + self._Hf

    def Ha(self, T: torch.Tensor | float) -> torch.Tensor:
        """Absolute specific enthalpy (same as H).

        Args:
            T: Temperature (K).

        Returns:
            Absolute specific enthalpy (J/kg).
        """
        return self.H(T)

    def Hs(self, T: torch.Tensor | float) -> torch.Tensor:
        """Sensible specific enthalpy (H without Hf).

        Args:
            T: Temperature (K).

        Returns:
            Sensible specific enthalpy (J/kg).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1.0)

        if abs(self._n + 1.0) < 1e-12:
            return self._Cp0 * torch.log(T_safe)

        return self._Cp0 / (self._n + 1.0) * T_safe.pow(self._n + 1.0)

    def E(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific internal energy: E = H - R*T.

        Args:
            T: Temperature (K).

        Returns:
            Specific internal energy (J/kg).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T_tensor = torch.tensor(T, dtype=dtype, device=device)
        else:
            T_tensor = T

        return self.H(T_tensor) - self._R * T_tensor

    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""
        return self._R

    @property
    def Cp0(self) -> float:
        """Reference specific heat coefficient."""
        return self._Cp0

    @property
    def exponent(self) -> float:
        """Power-law exponent."""
        return self._n

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
            f"HPowerThermo(R={self._R}, Cp0={self._Cp0}, "
            f"n={self._n}, Hf={self._Hf})"
        )
