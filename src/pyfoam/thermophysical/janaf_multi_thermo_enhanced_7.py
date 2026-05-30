"""
Enhanced JANAF thermodynamic model v7 — multi-reaction equilibrium and Cp extrapolation.

Extends :class:`~pyfoam.thermophysical.janaf_multi_thermo_enhanced_6.JanafMultiThermoEnhanced6`
with:

- Multi-reaction equilibrium constant computation (van't Hoff integration)
- High-temperature Cp extrapolation beyond JANAF range (T > 6000 K)
- Mixture-averaged Cp with mass-fraction weighting

Usage::

    from pyfoam.thermophysical.janaf_multi_thermo_enhanced_7 import JanafMultiThermoEnhanced7
    from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

    phases = [
        JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
        JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
    ]
    thermo = JanafMultiThermoEnhanced7(R=461.5, phases=phases, blend_width=5.0)
    K_eq = thermo.equilibrium_constant(300.0, Delta_H_rxn=-44e3, Delta_S_rxn=-120.0)
"""

from __future__ import annotations

import math
import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_6 import JanafMultiThermoEnhanced6
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

__all__ = ["JanafMultiThermoEnhanced7"]

logger = logging.getLogger(__name__)

_R_UNIV = 8.314462618  # J/(mol*K)


class JanafMultiThermoEnhanced7(JanafMultiThermoEnhanced6):
    """Enhanced multi-phase JANAF v7 with reaction equilibrium and Cp extrapolation.

    Extends :class:`JanafMultiThermoEnhanced6` with:

    - **Multi-reaction equilibrium**: computes K_eq(T) via van't Hoff integration
      from a reference temperature and reaction thermochemistry.
    - **High-temperature Cp extrapolation**: when T exceeds the JANAF table range,
      uses the last-phase polynomial with a logarithmic tail correction.
    - **Mixture-averaged Cp**: computes mass-fraction-weighted Cp for multi-species
      mixtures using an array of thermo models.

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
    S_ref : float
        Reference entropy at T_ref (J/(kg*K)). Default 0.
    T_ref : float
        Reference temperature for entropy (K). Default 298.15.
    S_ref_table : dict[float, float] or None
        Optional table of {T: S_ref(T)} for temperature-dependent
        reference entropy interpolation.
    Cp_extrapolate_power : float
        Power-law exponent for high-T Cp extrapolation. Default 0.5.
    T_extrapolate_start : float
        Temperature above which extrapolation activates (K). Default 6000.
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
        S_ref: float = 0.0,
        T_ref: float = 298.15,
        S_ref_table: dict[float, float] | None = None,
        Cp_extrapolate_power: float = 0.5,
        T_extrapolate_start: float = 6000.0,
    ) -> None:
        super().__init__(
            R=R, phases=phases, Hf=Hf, blend_width=blend_width,
            steepness=steepness, beta_coeffs=beta_coeffs,
            latent_sensible_fraction=latent_sensible_fraction,
            S_ref=S_ref, T_ref=T_ref, S_ref_table=S_ref_table,
        )
        self._Cp_extrap_power = Cp_extrapolate_power
        self._T_extrap_start = T_extrapolate_start

    @property
    def Cp_extrapolate_power(self) -> float:
        """Power-law exponent for high-T extrapolation."""
        return self._Cp_extrap_power

    @property
    def T_extrapolate_start(self) -> float:
        """Temperature threshold for extrapolation (K)."""
        return self._T_extrap_start

    # ------------------------------------------------------------------
    # Multi-reaction equilibrium constant (van't Hoff)
    # ------------------------------------------------------------------

    def equilibrium_constant(
        self,
        T: float,
        Delta_H_rxn: float,
        Delta_S_rxn: float,
        T_ref_k: float = 298.15,
    ) -> float:
        """Compute equilibrium constant K_eq via van't Hoff integration.

        ln(K_eq(T)) = ln(K_eq(T_ref)) - Delta_H_rxn/R * (1/T - 1/T_ref)

        where K_eq(T_ref) = exp(Delta_S_rxn/R - Delta_H_rxn/(R*T_ref)).

        Parameters
        ----------
        T : float
            Temperature (K).
        Delta_H_rxn : float
            Reaction enthalpy (J/mol).
        Delta_S_rxn : float
            Reaction entropy (J/(mol*K)).
        T_ref_k : float
            Reference temperature (K). Default 298.15.

        Returns
        -------
        float
            Equilibrium constant K_eq.
        """
        T = max(T, 1.0)
        ln_K_ref = Delta_S_rxn / _R_UNIV - Delta_H_rxn / (_R_UNIV * T_ref_k)
        ln_K = ln_K_ref - Delta_H_rxn / _R_UNIV * (1.0 / T - 1.0 / T_ref_k)
        # Clamp to prevent overflow
        ln_K = max(min(ln_K, 500.0), -500.0)
        return math.exp(ln_K)

    # ------------------------------------------------------------------
    # High-temperature Cp extrapolation
    # ------------------------------------------------------------------

    def _Cp_extrapolated(self, T: float) -> float:
        """Cp with high-temperature extrapolation beyond JANAF range.

        For T > T_extrapolate_start, uses:
            Cp(T) = Cp_last * (T / T_start)^power

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Extrapolated Cp (J/(kg*K)).
        """
        if T <= self._T_extrap_start:
            return float(self.Cp(T).item()) if hasattr(self.Cp(T), "item") else float(self.Cp(T))

        cp_ref = float(self.Cp(self._T_extrap_start).item()) if hasattr(
            self.Cp(self._T_extrap_start), "item"
        ) else float(self.Cp(self._T_extrap_start))
        ratio = T / max(self._T_extrap_start, 1.0)
        return cp_ref * ratio ** self._Cp_extrap_power

    # ------------------------------------------------------------------
    # Mixture-averaged Cp
    # ------------------------------------------------------------------

    def mixture_Cp(
        self,
        T: float,
        Y: Sequence[float],
        thermo_models: Sequence["JanafMultiThermoEnhanced7"],
    ) -> float:
        """Mass-fraction-weighted mixture Cp.

        Cp_mix = sum_i Y_i * Cp_i(T)

        Parameters
        ----------
        T : float
            Temperature (K).
        Y : sequence of float
            Mass fractions (must sum to 1).
        thermo_models : sequence of JanafMultiThermoEnhanced7
            One thermo model per species.

        Returns
        -------
        float
            Mixture Cp (J/(kg*K)).
        """
        cp_mix = 0.0
        for y_i, model in zip(Y, thermo_models):
            cp_i = float(model.Cp(T).item()) if hasattr(model.Cp(T), "item") else float(model.Cp(T))
            cp_mix += y_i * cp_i
        return cp_mix

    def __repr__(self) -> str:
        total_L = self.total_latent_heat()
        has_table = "table" if self._S_ref_table is not None else "const"
        return (
            f"JanafMultiThermoEnhanced7(R={self._R}, n_phases={len(self._phases)}, "
            f"Hf={self._Hf}, blend_width={self._blend_width}, "
            f"S_ref_mode={has_table}, L_total={total_L:.0f}, "
            f"Cp_extrap_power={self._Cp_extrap_power})"
        )
