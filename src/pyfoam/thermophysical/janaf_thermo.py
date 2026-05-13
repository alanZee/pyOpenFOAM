"""
JANAF polynomial thermodynamic model for specific heat capacity.

Implements the JANAF (Joint Army-Navy-Air Force) polynomial representation
of thermodynamic properties, as used in OpenFOAM's ``janafThermo`` class.

The specific heat capacity is represented as a polynomial in temperature:

.. math::

    C_p/R = a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4

where R is the specific gas constant. Enthalpy and internal energy are
obtained by integration:

.. math::

    H(T) = R T \\left(a_0 + \\frac{a_1 T}{2} + \\frac{a_2 T^2}{3}
           + \\frac{a_3 T^3}{4} + \\frac{a_4 T^4}{5}\\right) + H_f

.. math::

    E(T) = H(T) - R T

The model is valid within a temperature range ``[T_low, T_high]`` and
clamps input temperatures to this range.

Usage::

    from pyfoam.thermophysical.janaf_thermo import JanafThermo

    # Nitrogen (N2) JANAF coefficients (Cp/R, dimensionless)
    n2 = JanafThermo(
        R=296.8,
        coeffs=[3.53101, -0.000123661, -5.02999e-7, 2.43531e-9, -1.40881e-12],
    )
    cp = n2.Cp(T=300.0)  # ~1040.7 J/(kg·K)
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["JanafThermo"]

logger = logging.getLogger(__name__)


class JanafThermo:
    """JANAF polynomial thermodynamic model.

    Computes specific heat capacity, enthalpy, and internal energy using
    the JANAF polynomial representation. Coefficients are dimensionless
    (Cp/R form).

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg·K)).
    coeffs : sequence of float
        JANAF coefficients ``[a0, a1, a2, a3, a4]`` for
        ``Cp/R = a0 + a1*T + a2*T² + a3*T³ + a4*T⁴``.
    Hf : float
        Heat of formation (J/kg) at reference temperature. Default 0.
    T_low : float
        Lower temperature bound (K). Default 200.
    T_high : float
        Upper temperature bound (K). Default 6000.

    Examples::

        # Air (approximately constant Cp)
        air = JanafThermo(R=287.0, coeffs=[3.5])
        cp = air.Cp(T=300.0)  # ~1004.5 J/(kg·K)

        # Nitrogen (N2) with full JANAF data
        n2 = JanafThermo(
            R=296.8,
            coeffs=[3.53101, -0.000123661, -5.02999e-7, 2.43531e-9, -1.40881e-12],
        )
    """

    def __init__(
        self,
        R: float,
        coeffs: Sequence[float],
        Hf: float = 0.0,
        T_low: float = 200.0,
        T_high: float = 6000.0,
    ) -> None:
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if len(coeffs) == 0:
            raise ValueError("coeffs must not be empty")
        if len(coeffs) > 5:
            raise ValueError(f"coeffs must have at most 5 elements, got {len(coeffs)}")
        if T_low >= T_high:
            raise ValueError(f"T_low ({T_low}) must be < T_high ({T_high})")

        self._R = R
        # Pad coefficients to exactly 5 elements
        self._coeffs = list(coeffs) + [0.0] * (5 - len(coeffs))
        self._Hf = Hf
        self._T_low = T_low
        self._T_high = T_high

    def _clamp_T(self, T: torch.Tensor) -> torch.Tensor:
        """Clamp temperature to valid range."""
        return T.clamp(min=self._T_low, max=self._T_high)

    def _to_tensor(self, T: torch.Tensor | float) -> torch.Tensor:
        """Convert scalar to tensor if needed."""
        if isinstance(T, torch.Tensor):
            return T
        device = get_device()
        dtype = get_default_dtype()
        return torch.tensor(T, dtype=dtype, device=device)

    def Cp(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific heat capacity at constant pressure.

        .. math::

            C_p = R \\cdot (a_0 + a_1 T + a_2 T^2 + a_3 T^3 + a_4 T^4)

        Args:
            T: Temperature (K).

        Returns:
            Specific heat capacity (J/(kg·K)).
        """
        T = self._clamp_T(self._to_tensor(T))
        a0, a1, a2, a3, a4 = self._coeffs
        return self._R * (a0 + T * (a1 + T * (a2 + T * (a3 + T * a4))))

    def Cv(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific heat capacity at constant volume.

        .. math::

            C_v = C_p - R

        Args:
            T: Temperature (K).

        Returns:
            Specific heat capacity (J/(kg·K)).
        """
        return self.Cp(T) - self._R

    def gamma(self, T: torch.Tensor | float) -> torch.Tensor:
        """Ratio of specific heats (Cp/Cv).

        Args:
            T: Temperature (K).

        Returns:
            Ratio of specific heats (dimensionless).
        """
        cp = self.Cp(T)
        cv = cp - self._R
        return cp / cv

    def H(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific enthalpy.

        Obtained by integrating Cp from 0 to T:

        .. math::

            H(T) = R T \\left(a_0 + \\frac{a_1 T}{2} + \\frac{a_2 T^2}{3}
                   + \\frac{a_3 T^3}{4} + \\frac{a_4 T^4}{5}\\right) + H_f

        Args:
            T: Temperature (K).

        Returns:
            Specific enthalpy (J/kg).
        """
        T = self._clamp_T(self._to_tensor(T))
        a0, a1, a2, a3, a4 = self._coeffs
        return self._R * T * (
            a0
            + a1 * T / 2.0
            + a2 * T**2 / 3.0
            + a3 * T**3 / 4.0
            + a4 * T**4 / 5.0
        ) + self._Hf

    def E(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific internal energy.

        .. math::

            E(T) = H(T) - R T

        Args:
            T: Temperature (K).

        Returns:
            Specific internal energy (J/kg).
        """
        T_tensor = self._clamp_T(self._to_tensor(T))
        return self.H(T_tensor) - self._R * T_tensor

    def Ha(self, T: torch.Tensor | float) -> torch.Tensor:
        """Absolute specific enthalpy (alias for H).

        Args:
            T: Temperature (K).

        Returns:
            Absolute specific enthalpy (J/kg).
        """
        return self.H(T)

    def Hs(self, T: torch.Tensor | float) -> torch.Tensor:
        """Sensible specific enthalpy (H without heat of formation).

        Args:
            T: Temperature (K).

        Returns:
            Sensible specific enthalpy (J/kg).
        """
        T = self._clamp_T(self._to_tensor(T))
        a0, a1, a2, a3, a4 = self._coeffs
        return self._R * T * (
            a0
            + a1 * T / 2.0
            + a2 * T**2 / 3.0
            + a3 * T**3 / 4.0
            + a4 * T**4 / 5.0
        )

    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""
        return self._R

    @property
    def coeffs(self) -> list[float]:
        """JANAF coefficients [a0, a1, a2, a3, a4]."""
        return self._coeffs.copy()

    @property
    def Hf(self) -> float:
        """Heat of formation (J/kg)."""
        return self._Hf

    @property
    def T_low(self) -> float:
        """Lower temperature bound (K)."""
        return self._T_low

    @property
    def T_high(self) -> float:
        """Upper temperature bound (K)."""
        return self._T_high

    def __repr__(self) -> str:
        return (
            f"JanafThermo(R={self._R}, coeffs={self._coeffs}, "
            f"Hf={self._Hf}, T_low={self._T_low}, T_high={self._T_high})"
        )
