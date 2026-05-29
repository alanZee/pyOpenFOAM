"""
Multi-phase JANAF thermodynamic model.

Implements a multi-phase JANAF thermodynamic model that supports
piecewise polynomial Cp across different temperature ranges (phases),
as used in OpenFOAM's ``janafMultiThermo`` class.

Each phase has its own set of JANAF polynomial coefficients valid
within a specific temperature range. Properties are computed by
selecting the appropriate phase for the given temperature.

Usage::

    from pyfoam.thermophysical.janaf_multi_thermo import JanafMultiThermo, JanafPhase

    # Two-phase JANAF for water: liquid below 373.15 K, gas above
    phases = [
        JanafPhase(coeffs=[-2.0, 1.0, 0.0, 0.0, 0.0], T_low=200, T_high=373.15),
        JanafPhase(coeffs=[3.5, 0.001, 0.0, 0.0, 0.0], T_low=373.15, T_high=6000),
    ]
    thermo = JanafMultiThermo(R=461.5, phases=phases, Hf=-1.34e7)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["JanafMultiThermo", "JanafPhase"]

logger = logging.getLogger(__name__)


@dataclass
class JanafPhase:
    """A single JANAF phase with polynomial coefficients and temperature range.

    Parameters
    ----------
    coeffs : list of float
        JANAF coefficients ``[a0, a1, a2, a3, a4]`` for
        ``Cp/R = a0 + a1*T + a2*T² + a3*T³ + a4*T⁴``.
    T_low : float
        Lower temperature bound (K) for this phase.
    T_high : float
        Upper temperature bound (K) for this phase.
    Hf : float or None
        Phase-specific heat of formation (J/kg). If None, uses global Hf.
    L : float
        Latent heat at phase transition (J/kg). Default 0.
    """

    coeffs: list[float]
    T_low: float
    T_high: float
    Hf: float | None = None
    L: float = 0.0

    def __post_init__(self) -> None:
        if len(self.coeffs) == 0:
            raise ValueError("coeffs must not be empty")
        if len(self.coeffs) > 5:
            raise ValueError(f"coeffs must have at most 5 elements, got {len(self.coeffs)}")
        if self.T_low >= self.T_high:
            raise ValueError(f"T_low ({self.T_low}) must be < T_high ({self.T_high})")
        # Pad to 5 coefficients
        self.coeffs = list(self.coeffs) + [0.0] * (5 - len(self.coeffs))


class JanafMultiThermo:
    """Multi-phase JANAF thermodynamic model.

    Supports piecewise polynomial Cp across different temperature
    ranges (phases). Each phase can have different coefficients and
    a latent heat contribution.

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg·K)).
    phases : sequence of JanafPhase
        List of JANAF phases, ordered by temperature range.
        Ranges must be contiguous or overlap at boundaries.
    Hf : float
        Global heat of formation (J/kg). Default 0.
        Phase-specific Hf overrides this if set.

    Examples::

        phases = [
            JanafPhase(coeffs=[3.5], T_low=200, T_high=1000),
            JanafPhase(coeffs=[3.0, 5e-4], T_low=1000, T_high=6000),
        ]
        thermo = JanafMultiThermo(R=287.0, phases=phases)
        cp = thermo.Cp(T=300.0)
    """

    def __init__(
        self,
        R: float,
        phases: Sequence[JanafPhase],
        Hf: float = 0.0,
    ) -> None:
        if R <= 0:
            raise ValueError(f"R must be positive, got {R}")
        if len(phases) == 0:
            raise ValueError("phases must not be empty")

        self._R = R
        self._Hf = Hf
        self._phases = list(phases)

        # Compute cumulative latent heat offsets
        self._cumulative_L = [0.0]
        for phase in self._phases[:-1]:
            self._cumulative_L.append(self._cumulative_L[-1] + phase.L)

    def _find_phase(self, T: float | torch.Tensor) -> tuple[int, float]:
        """Find the phase index for a given temperature.

        For tensor input, returns the phase index for a scalar
        (used per-element in Cp, H, etc.).

        Args:
            T: Temperature (K).

        Returns:
            Tuple of ``(phase_index, T_clamped)``.
        """
        T_val = float(T) if isinstance(T, torch.Tensor) else float(T)
        for i, phase in enumerate(self._phases):
            if T_val <= phase.T_high or i == len(self._phases) - 1:
                return i, max(T_val, phase.T_low)
        return len(self._phases) - 1, T_val

    def _to_tensor(self, T: torch.Tensor | float) -> torch.Tensor:
        """Convert scalar to tensor if needed."""
        if isinstance(T, torch.Tensor):
            return T
        device = get_device()
        dtype = get_default_dtype()
        return torch.tensor(T, dtype=dtype, device=device)

    def Cp(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific heat capacity at constant pressure.

        Selects the appropriate phase and evaluates the polynomial.

        Args:
            T: Temperature (K).

        Returns:
            Specific heat capacity (J/(kg·K)).
        """
        T_tensor = self._to_tensor(T)

        # Handle scalar
        if T_tensor.dim() == 0:
            idx, T_clamped = self._find_phase(T_tensor)
            phase = self._phases[idx]
            T_c = torch.tensor(T_clamped, dtype=T_tensor.dtype, device=T_tensor.device)
            a0, a1, a2, a3, a4 = phase.coeffs
            return self._R * (a0 + T_c * (a1 + T_c * (a2 + T_c * (a3 + T_c * a4))))

        # Handle tensor
        result = torch.zeros_like(T_tensor)
        for i, phase in enumerate(self._phases):
            mask = (T_tensor >= phase.T_low) & (
                (T_tensor <= phase.T_high) | (i == len(self._phases) - 1)
            )
            if mask.any():
                T_c = T_tensor[mask].clamp(min=phase.T_low, max=phase.T_high)
                a0, a1, a2, a3, a4 = phase.coeffs
                result[mask] = self._R * (
                    a0 + T_c * (a1 + T_c * (a2 + T_c * (a3 + T_c * a4)))
                )
        return result

    def Cv(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific heat capacity at constant volume: Cv = Cp - R.

        Args:
            T: Temperature (K).

        Returns:
            Specific heat capacity (J/(kg·K)).
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
        """Specific enthalpy.

        Integrates Cp from 0 to T within the selected phase,
        adding cumulative latent heat for transitions.

        Args:
            T: Temperature (K).

        Returns:
            Specific enthalpy (J/kg).
        """
        T_tensor = self._to_tensor(T)

        if T_tensor.dim() == 0:
            idx, T_clamped = self._find_phase(T_tensor)
            phase = self._phases[idx]
            hf = phase.Hf if phase.Hf is not None else self._Hf
            T_c = torch.tensor(T_clamped, dtype=T_tensor.dtype, device=T_tensor.device)
            a0, a1, a2, a3, a4 = phase.coeffs
            h = self._R * T_c * (
                a0 + a1 * T_c / 2.0 + a2 * T_c**2 / 3.0
                + a3 * T_c**3 / 4.0 + a4 * T_c**4 / 5.0
            ) + hf + self._cumulative_L[idx]
            return h

        result = torch.zeros_like(T_tensor)
        for i, phase in enumerate(self._phases):
            mask = (T_tensor >= phase.T_low) & (
                (T_tensor <= phase.T_high) | (i == len(self._phases) - 1)
            )
            if mask.any():
                hf = phase.Hf if phase.Hf is not None else self._Hf
                T_c = T_tensor[mask].clamp(min=phase.T_low, max=phase.T_high)
                a0, a1, a2, a3, a4 = phase.coeffs
                result[mask] = self._R * T_c * (
                    a0 + a1 * T_c / 2.0 + a2 * T_c**2 / 3.0
                    + a3 * T_c**3 / 4.0 + a4 * T_c**4 / 5.0
                ) + hf + self._cumulative_L[i]
        return result

    def E(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific internal energy: E = H - R*T.

        Args:
            T: Temperature (K).

        Returns:
            Specific internal energy (J/kg).
        """
        T_tensor = self._to_tensor(T)
        return self.H(T_tensor) - self._R * T_tensor

    def Ha(self, T: torch.Tensor | float) -> torch.Tensor:
        """Absolute specific enthalpy (alias for H)."""
        return self.H(T)

    def Hs(self, T: torch.Tensor | float) -> torch.Tensor:
        """Sensible specific enthalpy (H without heat of formation).

        Args:
            T: Temperature (K).

        Returns:
            Sensible specific enthalpy (J/kg).
        """
        T_tensor = self._to_tensor(T)

        if T_tensor.dim() == 0:
            idx, T_clamped = self._find_phase(T_tensor)
            phase = self._phases[idx]
            T_c = torch.tensor(T_clamped, dtype=T_tensor.dtype, device=T_tensor.device)
            a0, a1, a2, a3, a4 = phase.coeffs
            return self._R * T_c * (
                a0 + a1 * T_c / 2.0 + a2 * T_c**2 / 3.0
                + a3 * T_c**3 / 4.0 + a4 * T_c**4 / 5.0
            ) + self._cumulative_L[idx]

        result = torch.zeros_like(T_tensor)
        for i, phase in enumerate(self._phases):
            mask = (T_tensor >= phase.T_low) & (
                (T_tensor <= phase.T_high) | (i == len(self._phases) - 1)
            )
            if mask.any():
                T_c = T_tensor[mask].clamp(min=phase.T_low, max=phase.T_high)
                a0, a1, a2, a3, a4 = phase.coeffs
                result[mask] = self._R * T_c * (
                    a0 + a1 * T_c / 2.0 + a2 * T_c**2 / 3.0
                    + a3 * T_c**3 / 4.0 + a4 * T_c**4 / 5.0
                ) + self._cumulative_L[i]
        return result

    def R(self) -> float:
        """Specific gas constant (J/(kg·K))."""
        return self._R

    @property
    def phases(self) -> list[JanafPhase]:
        """List of JANAF phases."""
        return self._phases.copy()

    @property
    def Hf(self) -> float:
        """Global heat of formation (J/kg)."""
        return self._Hf

    @property
    def n_phases(self) -> int:
        """Number of phases."""
        return len(self._phases)

    def __repr__(self) -> str:
        return (
            f"JanafMultiThermo(R={self._R}, n_phases={len(self._phases)}, "
            f"Hf={self._Hf})"
        )
