"""
Enhanced multi-phase JANAF thermodynamic model v3.

Extends :class:`~pyfoam.thermophysical.janaf_multi_thermo_enhanced_2.JanafMultiThermoEnhanced2`
with improved capabilities:

- Multi-boundary latent heat with cumulative blending
- Pressure-dependent Cp via extended departure functions
- Gibbs free energy computation for phase equilibrium
- Configurable sigmoid steepness for phase transitions

Usage::

    from pyfoam.thermophysical.janaf_multi_thermo_enhanced_3 import JanafMultiThermoEnhanced3
    from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

    phases = [
        JanafPhase(coeffs=[3.5], T_low=200, T_high=373.15, L=2.26e6),
        JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
    ]
    thermo = JanafMultiThermoEnhanced3(R=461.5, phases=phases, blend_width=5.0)
    G = thermo.gibbs_free_energy(300.0)
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_2 import JanafMultiThermoEnhanced2
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

__all__ = ["JanafMultiThermoEnhanced3"]

logger = logging.getLogger(__name__)


class JanafMultiThermoEnhanced3(JanafMultiThermoEnhanced2):
    """Enhanced multi-phase JANAF v3 with cumulative latent heat and Gibbs energy.

    Extends :class:`JanafMultiThermoEnhanced2` with:

    - **Cumulative latent heat blending**: all latent heats accumulate across
      multiple phase transitions, giving correct enthalpy jumps at every boundary.
    - **Extended pressure departure**: multi-order correction with configurable
      beta coefficients for real-gas Cp(T, P).
    - **Gibbs free energy**: G = H - T*S using entropy integration from Cp/T.
    - **Sigmoid steepness**: configurable steepness parameter (n) for the phase
      transition sigmoid: alpha = 1 / (1 + exp(-2n * (T - T_b) / w)).

    Parameters
    ----------
    R : float
        Specific gas constant (J/(kg*K)).
    phases : sequence of JanafPhase
        JANAF phases ordered by temperature.
    Hf : float
        Global heat of formation (J/kg). Default 0.
    blend_width : float
        Temperature width (K) for phase-boundary blending. Default 0.
    steepness : float
        Sigmoid steepness exponent. Default 2.0 (standard sigmoid).
    beta_coeffs : sequence of float or None
        Pressure departure coefficients [beta1, beta2, ...].
        Cp_dep = Cp * (1 + beta1*dP/P_ref + beta2*(dP/P_ref)^2 + ...).
        Default None (uses single beta=1e-6).
    """

    def __init__(
        self,
        R: float,
        phases: Sequence[JanafPhase],
        Hf: float = 0.0,
        blend_width: float = 0.0,
        steepness: float = 2.0,
        beta_coeffs: Sequence[float] | None = None,
    ) -> None:
        super().__init__(R=R, phases=phases, Hf=Hf, blend_width=blend_width)
        self._steepness = steepness
        self._beta_coeffs = list(beta_coeffs) if beta_coeffs else [1e-6]

    @property
    def steepness(self) -> float:
        """Sigmoid steepness parameter."""
        return self._steepness

    @property
    def beta_coeffs(self) -> list[float]:
        """Pressure departure coefficients."""
        return self._beta_coeffs.copy()

    # ------------------------------------------------------------------
    # Override blend weight for configurable steepness
    # ------------------------------------------------------------------

    def _blend_weight_steep(self, T: float, T_boundary: float) -> float:
        """Blend weight with configurable steepness.

        w = 1 / (1 + exp(-2 * steepness * (T - T_boundary) / blend_width))
        """
        if self._blend_width <= 0:
            return 1.0 if T >= T_boundary else 0.0
        arg = -2.0 * self._steepness * (T - T_boundary) / self._blend_width
        arg = max(min(arg, 500.0), -500.0)
        return 1.0 / (1.0 + math.exp(arg))

    # ------------------------------------------------------------------
    # Enhanced pressure departure (multi-order)
    # ------------------------------------------------------------------

    def Cp_departure(
        self,
        T: float,
        P: float,
        P_ref: float = 101325.0,
    ) -> float:
        """Multi-order pressure departure correction to Cp.

        Cp_dep = Cp(T) * (1 + beta1*dP/P_ref + beta2*(dP/P_ref)^2 + ...)

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
            Corrected Cp (J/(kg*K)).
        """
        cp = float(self.Cp(T).item())
        dP_ratio = (P - P_ref) / max(P_ref, 1.0)
        correction = 1.0
        for i, beta in enumerate(self._beta_coeffs):
            correction += beta * dP_ratio ** (i + 1)
        return cp * correction

    # ------------------------------------------------------------------
    # Gibbs free energy
    # ------------------------------------------------------------------

    def entropy(self, T: float) -> float:
        """Approximate specific entropy via S = integral(Cp/T dT).

        Uses trapezoidal integration from T_low of the first phase to T.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Specific entropy (J/(kg*K)).
        """
        T_low = self._phases[0].T_low
        T_high = self._phases[-1].T_high
        T_clamped = max(T_low + 1.0, min(T, T_high))

        n_steps = 100
        dT = (T_clamped - T_low) / n_steps
        S = 0.0
        for i in range(n_steps):
            T1 = T_low + i * dT
            T2 = T_low + (i + 1) * dT
            cp1 = float(self.Cp(T1).item())
            cp2 = float(self.Cp(T2).item())
            T1_safe = max(T1, 1e-10)
            T2_safe = max(T2, 1e-10)
            S += 0.5 * (cp1 / T1_safe + cp2 / T2_safe) * dT

        return S

    def gibbs_free_energy(self, T: float) -> float:
        """Compute Gibbs free energy: G = H - T*S.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Specific Gibbs free energy (J/kg).
        """
        h = float(self.H(T).item())
        s = self.entropy(T)
        return h - T * s

    # ------------------------------------------------------------------
    # Cumulative latent heat blending
    # ------------------------------------------------------------------

    def total_latent_heat_up_to(self, transition_index: int) -> float:
        """Cumulative latent heat up to and including the given transition.

        Parameters
        ----------
        transition_index : int
            Transition index (inclusive upper bound).

        Returns
        -------
        float
            Cumulative latent heat (J/kg).

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
        return sum(self._latent_heats[: transition_index + 1])

    def __repr__(self) -> str:
        total_L = self.total_latent_heat()
        return (
            f"JanafMultiThermoEnhanced3(R={self._R}, n_phases={len(self._phases)}, "
            f"Hf={self._Hf}, blend_width={self._blend_width}, "
            f"steepness={self._steepness}, L_total={total_L:.0f})"
        )
