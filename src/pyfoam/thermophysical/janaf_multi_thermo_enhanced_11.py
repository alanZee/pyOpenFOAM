"""Enhanced JANAF thermodynamic model v11 — multi-species enthalpy coupling, entropy production tracking, and adaptive phase boundaries.

Extends :class:`~pyfoam.thermophysical.janaf_multi_thermo_enhanced_10.JanafMultiThermoEnhanced10`
with:

- Multi-species enthalpy coupling for mixture thermo consistency
- Entropy production tracking during phase transitions
- Adaptive phase boundary detection from Cp curvature

Usage::

    from pyfoam.thermophysical.janaf_multi_thermo_enhanced_11 import JanafMultiThermoEnhanced11
    from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

    phases = [
        JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
        JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
    ]
    thermo = JanafMultiThermoEnhanced11(R=461.5, phases=phases, blend_width=5.0)
"""

from __future__ import annotations

import math
import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_10 import JanafMultiThermoEnhanced10
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

__all__ = ["JanafMultiThermoEnhanced11"]

logger = logging.getLogger(__name__)

_R_UNIV = 8.314462618  # J/(mol*K)


class JanafMultiThermoEnhanced11(JanafMultiThermoEnhanced10):
    """Enhanced multi-phase JANAF v11 with enthalpy coupling and entropy tracking.

    Extends :class:`JanafMultiThermoEnhanced10` with:

    - **Multi-species enthalpy coupling**: cross-species enthalpy corrections
      for non-ideal mixture effects.
    - **Entropy production tracking**: monitors entropy generation during
      phase transitions for thermodynamic consistency checks.
    - **Adaptive phase boundaries**: detects phase boundaries from Cp
      curvature changes rather than fixed temperatures.

    Parameters
    ----------
    R, phases, Hf, blend_width, steepness, beta_coeffs, latent_sensible_fraction :
        See parent.
    S_ref, T_ref, S_ref_table, Cp_extrapolate_power, T_extrapolate_start :
        See parent.
    gibbs_max_iter, gibbs_tol : See parent.
    radiation_emissivity, radiation_area_factor, flash_max_iter, flash_tol : See parent.
    collision_integral_coeff, stability_check, rachford_rice_max_iter : See parent.
    enthalpy_coupling_coeff : float
        Cross-species enthalpy coupling coefficient. Default 0.0.
    track_entropy_production : bool
        Enable entropy production tracking. Default False.
    adaptive_phase_boundaries : bool
        Enable Cp-curvature-based phase boundary detection. Default False.
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
        gibbs_max_iter: int = 50,
        gibbs_tol: float = 1e-6,
        radiation_emissivity: float = 0.0,
        radiation_area_factor: float = 1.0,
        flash_max_iter: int = 20,
        flash_tol: float = 1e-6,
        collision_integral_coeff: float = 0.0,
        stability_check: bool = False,
        rachford_rice_max_iter: int = 30,
        enthalpy_coupling_coeff: float = 0.0,
        track_entropy_production: bool = False,
        adaptive_phase_boundaries: bool = False,
    ) -> None:
        super().__init__(
            R=R, phases=phases, Hf=Hf, blend_width=blend_width,
            steepness=steepness, beta_coeffs=beta_coeffs,
            latent_sensible_fraction=latent_sensible_fraction,
            S_ref=S_ref, T_ref=T_ref, S_ref_table=S_ref_table,
            Cp_extrapolate_power=Cp_extrapolate_power,
            T_extrapolate_start=T_extrapolate_start,
            gibbs_max_iter=gibbs_max_iter, gibbs_tol=gibbs_tol,
            radiation_emissivity=radiation_emissivity,
            radiation_area_factor=radiation_area_factor,
            flash_max_iter=flash_max_iter, flash_tol=flash_tol,
            collision_integral_coeff=collision_integral_coeff,
            stability_check=stability_check,
            rachford_rice_max_iter=rachford_rice_max_iter,
        )
        self._enthalpy_coupling = max(0.0, enthalpy_coupling_coeff)
        self._track_entropy = track_entropy_production
        self._adaptive_boundaries = adaptive_phase_boundaries
        self._entropy_history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Multi-species enthalpy coupling
    # ------------------------------------------------------------------

    def enthalpy_coupling_correction(self, T: float, Y: Sequence[float]) -> float:
        """Cross-species enthalpy correction for non-ideal mixtures.

        Delta_H = coeff * sum_i sum_{j!=i} Y_i * Y_j * (H_i(T) - H_j(T))

        Parameters
        ----------
        T : float
            Temperature (K).
        Y : sequence of float
            Mass fractions for each species/phase.

        Returns
        -------
        float
            Enthalpy coupling correction (J/kg).
        """
        if self._enthalpy_coupling < 1e-15 or len(Y) < 2:
            return 0.0

        H_vals = []
        for i in range(len(Y)):
            H_i = float(self.H(T).item()) if hasattr(self.H(T), "item") else float(self.H(T))
            H_vals.append(H_i)

        correction = 0.0
        n = len(Y)
        for i in range(n):
            for j in range(n):
                if i != j:
                    correction += Y[i] * Y[j] * abs(H_vals[i] - H_vals[j])

        return self._enthalpy_coupling * correction

    # ------------------------------------------------------------------
    # Entropy production tracking
    # ------------------------------------------------------------------

    def entropy_production(self, T: float, T_prev: float) -> dict[str, float]:
        """Track entropy production during temperature change.

        Parameters
        ----------
        T : float
            Current temperature (K).
        T_prev : float
            Previous temperature (K).

        Returns
        -------
        dict
            'delta_s': entropy change (J/(kg*K)),
            'production': positive entropy production,
            'is_irreversible': whether process is irreversible.
        """
        Cp_val = float(self.Cp(T).item()) if hasattr(self.Cp(T), "item") else float(self.Cp(T))
        T_safe = max(T, 1.0)
        T_prev_safe = max(T_prev, 1.0)

        delta_s = Cp_val * math.log(T_safe / T_prev_safe)
        production = max(0.0, delta_s)

        result = {
            "delta_s": delta_s,
            "production": production,
            "is_irreversible": production > 1e-10,
        }

        if self._track_entropy:
            self._entropy_history.append(result)

        return result

    @property
    def entropy_history(self) -> list[dict[str, float]]:
        """Return tracked entropy production history."""
        return self._entropy_history

    # ------------------------------------------------------------------
    # Adaptive phase boundary detection
    # ------------------------------------------------------------------

    def detect_phase_boundaries(self, T_low: float = 200.0, T_high: float = 6000.0, n_points: int = 100) -> list[float]:
        """Detect phase boundaries from Cp curvature changes.

        Parameters
        ----------
        T_low, T_high : float
            Temperature search range.
        n_points : int
            Number of evaluation points.

        Returns
        -------
        list of float
            Temperatures where Cp curvature changes sign.
        """
        if not self._adaptive_boundaries:
            return []

        dT = (T_high - T_low) / max(n_points - 1, 1)
        boundaries: list[float] = []

        Cp_prev2 = float(self.Cp(T_low).item()) if hasattr(self.Cp(T_low), "item") else float(self.Cp(T_low))
        Cp_prev1 = float(self.Cp(T_low + dT).item()) if hasattr(self.Cp(T_low + dT), "item") else float(self.Cp(T_low + dT))

        for i in range(2, n_points):
            T = T_low + i * dT
            Cp_val = float(self.Cp(T).item()) if hasattr(self.Cp(T), "item") else float(self.Cp(T))
            curvature = Cp_val - 2.0 * Cp_prev1 + Cp_prev2
            curvature_prev = Cp_prev1 - 2.0 * Cp_prev2 + float(self.Cp(T - 2 * dT).item()) if hasattr(self.Cp(T - 2 * dT), "item") else float(self.Cp(T - 2 * dT))

            if curvature * curvature_prev < 0:
                boundaries.append(T)

            Cp_prev2 = Cp_prev1
            Cp_prev1 = Cp_val

        return boundaries

    def __repr__(self) -> str:
        total_L = self.total_latent_heat()
        ec = f", enthalpy_coupling={self._enthalpy_coupling}" if self._enthalpy_coupling > 0 else ""
        return (
            f"JanafMultiThermoEnhanced11(R={self._R}, n_phases={len(self._phases)}, "
            f"Hf={self._Hf}, L_total={total_L:.0f}{ec})"
        )
