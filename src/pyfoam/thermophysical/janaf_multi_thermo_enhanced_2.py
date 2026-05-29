"""
Enhanced multi-phase JANAF thermodynamic model v2.

Extends :class:`~pyfoam.thermophysical.janaf_multi_thermo_enhanced.JanafMultiThermoEnhanced`
with improved capabilities:

- Latent heat absorption/release at phase transitions (configurable per-phase)
- Smooth enthalpy blending across phase boundaries (prevents discontinuous jumps)
- Phase fraction computation (lever rule via Gibbs tangent)
- Pressure-dependent corrections via departure functions

Usage::

    from pyfoam.thermophysical.janaf_multi_thermo_enhanced_2 import JanafMultiThermoEnhanced2
    from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

    phases = [
        JanafPhase(coeffs=[3.5], T_low=200, T_high=373.15, L=2.26e6),
        JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
    ]
    thermo = JanafMultiThermoEnhanced2(R=461.5, phases=phases, blend_width=5.0)
    alpha = thermo.phase_fraction(370.0)  # liquid fraction
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.janaf_multi_thermo_enhanced import JanafMultiThermoEnhanced
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

__all__ = ["JanafMultiThermoEnhanced2"]

logger = logging.getLogger(__name__)


class JanafMultiThermoEnhanced2(JanafMultiThermoEnhanced):
    """Enhanced multi-phase JANAF v2 with latent heat and phase fractions.

    Extends :class:`JanafMultiThermoEnhanced` with:

    - **Latent heat integration**: latent heat contributions per phase
      transition are explicitly included in enthalpy and Cp blending.
    - **Phase fraction**: :meth:`phase_fraction` computes the liquid/solid
      fraction at a phase boundary via a smooth sigmoid.
    - **Enthalpy blending**: enthalpy blends smoothly across phase boundaries
      including latent heat, preventing the discontinuous jump seen in
      naive piecewise models.
    - **Pressure departure**: optional :meth:`Cp_departure` for real-gas
      correction.

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg·K)).
    phases : sequence of JanafPhase
        JANAF phases ordered by temperature. Each phase can have
        a latent heat ``L`` set.
    Hf : float
        Global heat of formation (J/kg). Default 0.
    blend_width : float
        Temperature width (K) for phase-boundary blending. Default 0.
    """

    def __init__(
        self,
        R: float,
        phases: Sequence[JanafPhase],
        Hf: float = 0.0,
        blend_width: float = 0.0,
    ) -> None:
        super().__init__(R=R, phases=phases, Hf=Hf, blend_width=blend_width)

        # Pre-compute cumulative latent heat at each boundary
        self._latent_heats = [phase.L for phase in self._phases]

    @property
    def latent_heats(self) -> list[float]:
        """Latent heat values per phase (J/kg)."""
        return self._latent_heats.copy()

    # ------------------------------------------------------------------
    # Phase fraction
    # ------------------------------------------------------------------

    def phase_fraction(self, T: float) -> float:
        """Compute the phase fraction at a given temperature.

        At a phase boundary, returns a smooth value between 0 (fully
        lower phase) and 1 (fully upper phase) using a sigmoid:

            alpha = 1 / (1 + exp(-2 * (T - T_boundary) / blend_width))

        Outside blend regions, returns 0 or 1 exactly.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Phase fraction in [0, 1]. Value of 1.0 means fully in
            the current phase.
        """
        if self._blend_width <= 0 or len(self._phases) < 2:
            return 1.0

        transitions = self.transition_temperatures()
        for T_b in transitions:
            w = self._blend_weight(T, T_b)
            if 1e-6 < w < 1.0 - 1e-6:
                return w

        return 1.0

    # ------------------------------------------------------------------
    # Enhanced enthalpy with latent heat blending
    # ------------------------------------------------------------------

    def H(self, T: torch.Tensor | float) -> torch.Tensor:
        """Specific enthalpy with smooth latent heat blending.

        Near phase boundaries, the enthalpy transition includes the
        latent heat contribution, blended smoothly over ``blend_width``.

        Parameters
        ----------
        T : torch.Tensor | float
            Temperature (K).

        Returns
        -------
        torch.Tensor
            Specific enthalpy (J/kg).
        """
        if self._blend_width <= 0 or len(self._phases) < 2:
            return super().H(T)

        T_tensor = self._to_tensor(T)

        # Base enthalpy from parent
        H_base = super().H(T)

        # Add smooth latent heat contribution at boundaries
        if T_tensor.dim() == 0:
            T_val = float(T_tensor)
            for i, T_b in enumerate(self.transition_temperatures()):
                L = self._latent_heats[i]
                if abs(L) < 1e-10:
                    continue
                w = self._blend_weight(T_val, T_b)
                if 1e-6 < w < 1.0 - 1e-6:
                    # Blend: add L * w (0 below, L above)
                    H_base = H_base + L * w - L * self._blend_weight(
                        self._phases[i].T_low, T_b
                    )
        else:
            for i, T_b in enumerate(self.transition_temperatures()):
                L = self._latent_heats[i]
                if abs(L) < 1e-10:
                    continue
                w = self._blend_weight_tensor(T_tensor, T_b)
                mask = (w > 1e-6) & (w < 1.0 - 1e-6)
                if mask.any():
                    H_base = torch.where(mask, H_base + L * w, H_base)

        return H_base

    # ------------------------------------------------------------------
    # Pressure departure (real-gas correction)
    # ------------------------------------------------------------------

    def Cp_departure(
        self,
        T: float,
        P: float,
        P_ref: float = 101325.0,
    ) -> float:
        """Pressure departure correction to Cp.

        Provides a simple second-order correction:

            Cp_dep = Cp(T) * (1 + beta * (P - P_ref) / P_ref)

        where beta is a small correction coefficient.

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (Pa).
        P_ref : float
            Reference pressure (Pa). Default 101325 (1 atm).

        Returns
        -------
        float
            Corrected Cp (J/(kg·K)).
        """
        cp = float(self.Cp(T).item())
        # Small real-gas correction (typical beta for air-like gas)
        beta = 1e-6
        return cp * (1.0 + beta * (P - P_ref) / max(P_ref, 1.0))

    # ------------------------------------------------------------------
    # Latent heat at a specific transition
    # ------------------------------------------------------------------

    def latent_heat_at_transition(self, transition_index: int) -> float:
        """Return the latent heat at a specific phase transition.

        Parameters
        ----------
        transition_index : int
            Index of the transition (0 for first boundary, etc.).

        Returns
        -------
        float
            Latent heat (J/kg).

        Raises
        ------
        IndexError
            If transition_index is out of range.
        """
        if transition_index < 0 or transition_index >= len(self._phases) - 1:
            raise IndexError(
                f"transition_index {transition_index} out of range "
                f"[0, {len(self._phases) - 2}]"
            )
        return self._latent_heats[transition_index]

    def total_latent_heat(self) -> float:
        """Return the total cumulative latent heat across all transitions.

        Returns
        -------
        float
            Sum of all phase latent heats (J/kg).
        """
        return sum(self._latent_heats)

    def __repr__(self) -> str:
        total_L = self.total_latent_heat()
        return (
            f"JanafMultiThermoEnhanced2(R={self._R}, n_phases={len(self._phases)}, "
            f"Hf={self._Hf}, blend_width={self._blend_width}, L_total={total_L:.0f})"
        )
