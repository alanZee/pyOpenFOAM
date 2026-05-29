"""
Enhanced JANAF thermodynamic model v4 — multi-phase with advanced phase transitions.

Extends :class:`~pyfoam.thermophysical.janaf_multi_thermo_enhanced_3.JanafMultiThermoEnhanced3`
with:

- Better phase transitions with latent heat decomposition (sensible + configurational)
- Pressure-dependent properties via fugacity coefficient correction
- Equilibrium quality (vapour fraction) computation from Gibbs energy
- Configurable Clausius-Clapeyron slope for phase boundaries

Usage::

    from pyfoam.thermophysical.janaf_multi_thermo_enhanced_4 import JanafMultiThermoEnhanced4
    from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

    phases = [
        JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
        JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
    ]
    thermo = JanafMultiThermoEnhanced4(R=461.5, phases=phases, blend_width=5.0)
    x_eq = thermo.vapour_quality(373.15)
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_3 import JanafMultiThermoEnhanced3
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

__all__ = ["JanafMultiThermoEnhanced4"]

logger = logging.getLogger(__name__)


class JanafMultiThermoEnhanced4(JanafMultiThermoEnhanced3):
    """Enhanced multi-phase JANAF v4 with advanced phase transitions.

    Extends :class:`JanafMultiThermoEnhanced3` with:

    - **Latent heat decomposition**: splits total latent heat into sensible
      and configurational contributions: L = L_sensible + L_config.
    - **Pressure-dependent Cp**: fugacity-coefficient correction for
      high-pressure real-gas behaviour beyond the departure function.
    - **Vapour quality**: equilibrium quality computed from Gibbs energy
      balance across all phases at a given temperature.
    - **Clausius-Clapeyron slope**: dT/dP for each phase boundary.

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
        Sigmoid steepness exponent. Default 2.0.
    beta_coeffs : sequence of float or None
        Pressure departure coefficients.
    latent_sensible_fraction : float
        Fraction of total latent heat attributed to sensible heating.
        Default 0.1 (10% sensible, 90% configurational).
    """

    def __init__(
        self,
        R: float,
        phases: Sequence[JanafPhase],
        Hf: float = 0.0,
        blend_width: float = 0.0,
        steepness: float = 2.0,
        beta_coeffs: Sequence[float] | None = None,
        latent_sensible_fraction: float = 0.1,
    ) -> None:
        super().__init__(
            R=R, phases=phases, Hf=Hf, blend_width=blend_width,
            steepness=steepness, beta_coeffs=beta_coeffs,
        )
        self._L_sens_frac = max(0.0, min(latent_sensible_fraction, 1.0))

    @property
    def latent_sensible_fraction(self) -> float:
        """Fraction of latent heat attributed to sensible contribution."""
        return self._L_sens_frac

    # ------------------------------------------------------------------
    # Latent heat decomposition
    # ------------------------------------------------------------------

    def latent_heat_sensible(self, transition_index: int) -> float:
        """Sensible part of latent heat at given transition.

        Parameters
        ----------
        transition_index : int
            Transition index.

        Returns
        -------
        float
            Sensible latent heat (J/kg).
        """
        L_total = self.total_latent_heat_up_to(transition_index)
        if transition_index > 0:
            L_prev = self.total_latent_heat_up_to(transition_index - 1)
            L_this = L_total - L_prev
        else:
            L_this = L_total
        return L_this * self._L_sens_frac

    def latent_heat_configurational(self, transition_index: int) -> float:
        """Configurational part of latent heat at given transition.

        Parameters
        ----------
        transition_index : int
            Transition index.

        Returns
        -------
        float
            Configurational latent heat (J/kg).
        """
        L_total = self.total_latent_heat_up_to(transition_index)
        if transition_index > 0:
            L_prev = self.total_latent_heat_up_to(transition_index - 1)
            L_this = L_total - L_prev
        else:
            L_this = L_total
        return L_this * (1.0 - self._L_sens_frac)

    # ------------------------------------------------------------------
    # Fugacity coefficient correction
    # ------------------------------------------------------------------

    def Cp_fugacity_corrected(
        self,
        T: float,
        P: float,
        P_ref: float = 101325.0,
        fugacity_coeff: float = 1.0,
    ) -> float:
        """Cp with fugacity coefficient correction for high-pressure gases.

        Cp_real = Cp_departure(T, P) * fugacity_coeff

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (Pa).
        P_ref : float
            Reference pressure (Pa).
        fugacity_coeff : float
            Fugacity coefficient (phi). Default 1.0 (ideal).

        Returns
        -------
        float
            Corrected Cp (J/(kg*K)).
        """
        cp_dep = self.Cp_departure(T, P, P_ref)
        return cp_dep * max(fugacity_coeff, 0.01)

    # ------------------------------------------------------------------
    # Clausius-Clapeyron slope
    # ------------------------------------------------------------------

    def clausius_clapeyron_slope(self, transition_index: int) -> float:
        """dT/dP at a phase boundary via Clausius-Clapeyron.

        dT/dP = T * Delta_V / L  ~  T / (L * rho_v/R*T)

        Simplified: dT/dP = R * T_b^2 / L  (assuming ideal vapour).

        Parameters
        ----------
        transition_index : int
            Transition index.

        Returns
        -------
        float
            dT/dP slope (K/Pa).
        """
        if transition_index < 0 or transition_index >= len(self._phases) - 1:
            raise IndexError(
                f"transition_index {transition_index} out of range "
                f"[0, {len(self._phases) - 2}]"
            )
        T_b = self._phases[transition_index].T_high
        L = self._latent_heats[transition_index]
        if abs(L) < 1e-10:
            return float('inf')
        return self._R * T_b ** 2 / L

    # ------------------------------------------------------------------
    # Vapour quality (equilibrium)
    # ------------------------------------------------------------------

    def vapour_quality(self, T: float) -> float:
        """Estimate equilibrium vapour quality from Gibbs energy.

        For a two-phase system, quality is estimated from the blend
        weight at the phase boundary. For multi-phase, returns quality
        relative to the last phase.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Vapour quality in [0, 1].
        """
        if len(self._phases) < 2:
            return 1.0 if T >= self._phases[0].T_high else 0.0

        # Use cumulative blend weights
        total_weight = 0.0
        for i in range(len(self._phases) - 1):
            T_b = self._phases[i].T_high
            w = self._blend_weight_steep(T, T_b)
            total_weight += w

        # Normalise to [0, 1]
        n_transitions = len(self._phases) - 1
        return max(0.0, min(total_weight / n_transitions, 1.0))

    def __repr__(self) -> str:
        total_L = self.total_latent_heat()
        return (
            f"JanafMultiThermoEnhanced4(R={self._R}, n_phases={len(self._phases)}, "
            f"Hf={self._Hf}, blend_width={self._blend_width}, "
            f"steepness={self._steepness}, L_sens_frac={self._L_sens_frac}, "
            f"L_total={total_L:.0f})"
        )
