"""
Enhanced JANAF thermodynamic model v8 — multi-species equilibrium and mixture entropy.

Extends :class:`~pyfoam.thermophysical.janaf_multi_thermo_enhanced_7.JanafMultiThermoEnhanced7`
with:

- Multi-species chemical equilibrium via Gibbs minimisation
- Temperature-dependent reference entropy interpolation
- Mixture entropy computation with mass-fraction weighting

Usage::

    from pyfoam.thermophysical.janaf_multi_thermo_enhanced_8 import JanafMultiThermoEnhanced8
    from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

    phases = [
        JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
        JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
    ]
    thermo = JanafMultiThermoEnhanced8(R=461.5, phases=phases, blend_width=5.0)
"""

from __future__ import annotations

import math
import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_7 import JanafMultiThermoEnhanced7
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

__all__ = ["JanafMultiThermoEnhanced8"]

logger = logging.getLogger(__name__)

_R_UNIV = 8.314462618  # J/(mol*K)


class JanafMultiThermoEnhanced8(JanafMultiThermoEnhanced7):
    """Enhanced multi-phase JANAF v8 with Gibbs minimisation and mixture entropy.

    Extends :class:`JanafMultiThermoEnhanced7` with:

    - **Multi-species equilibrium**: iterative Gibbs free energy minimisation
      for multi-component mixtures at given T and p.
    - **Temperature-dependent S_ref interpolation**: uses the S_ref_table
      (if provided) for temperature-dependent reference entropy.
    - **Mixture entropy**: mass-fraction-weighted entropy computation for
      multi-species systems.

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
    gibbs_max_iter : int
        Maximum iterations for Gibbs minimisation. Default 50.
    gibbs_tol : float
        Convergence tolerance for Gibbs minimisation. Default 1e-6.
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
    ) -> None:
        super().__init__(
            R=R, phases=phases, Hf=Hf, blend_width=blend_width,
            steepness=steepness, beta_coeffs=beta_coeffs,
            latent_sensible_fraction=latent_sensible_fraction,
            S_ref=S_ref, T_ref=T_ref, S_ref_table=S_ref_table,
            Cp_extrapolate_power=Cp_extrapolate_power,
            T_extrapolate_start=T_extrapolate_start,
        )
        self._gibbs_max_iter = max(1, gibbs_max_iter)
        self._gibbs_tol = max(1e-12, gibbs_tol)

    @property
    def gibbs_max_iter(self) -> int:
        """Maximum Gibbs minimisation iterations."""
        return self._gibbs_max_iter

    # ------------------------------------------------------------------
    # Temperature-dependent reference entropy
    # ------------------------------------------------------------------

    def S_ref_at_T(self, T: float) -> float:
        """Reference entropy at temperature T.

        If an S_ref_table was provided, interpolates linearly; otherwise
        returns the constant S_ref.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Reference entropy (J/(kg*K)).
        """
        if self._S_ref_table is None or len(self._S_ref_table) < 2:
            return self._S_ref

        temps = sorted(self._S_ref_table.keys())
        if T <= temps[0]:
            return self._S_ref_table[temps[0]]
        if T >= temps[-1]:
            return self._S_ref_table[temps[-1]]

        # Linear interpolation
        for i in range(len(temps) - 1):
            T_lo, T_hi = temps[i], temps[i + 1]
            if T_lo <= T <= T_hi:
                frac = (T - T_lo) / max(T_hi - T_lo, 1e-30)
                S_lo = self._S_ref_table[T_lo]
                S_hi = self._S_ref_table[T_hi]
                return S_lo + frac * (S_hi - S_lo)

        return self._S_ref

    # ------------------------------------------------------------------
    # Mixture entropy
    # ------------------------------------------------------------------

    def mixture_entropy(
        self,
        T: float,
        Y: Sequence[float],
        thermo_models: Sequence["JanafMultiThermoEnhanced8"],
    ) -> float:
        """Mass-fraction-weighted mixture entropy.

        S_mix = sum_i Y_i * S_i(T)

        Parameters
        ----------
        T : float
            Temperature (K).
        Y : sequence of float
            Mass fractions (must sum to 1).
        thermo_models : sequence of JanafMultiThermoEnhanced8
            One thermo model per species.

        Returns
        -------
        float
            Mixture entropy (J/(kg*K)).
        """
        s_mix = 0.0
        for y_i, model in zip(Y, thermo_models):
            s_i = model.S_ref_at_T(T)
            # Add ideal gas entropy contribution: R * ln(T/T_ref)
            T_safe = max(T, 1.0)
            T_ref_safe = max(model._T_ref, 1.0)
            s_i += model._R * math.log(T_safe / T_ref_safe)
            s_mix += y_i * s_i
        return s_mix

    # ------------------------------------------------------------------
    # Gibbs free energy minimisation (simplified)
    # ------------------------------------------------------------------

    def gibbs_minimise(
        self,
        T: float,
        initial_Y: Sequence[float],
        thermo_models: Sequence["JanafMultiThermoEnhanced8"],
    ) -> list[float]:
        """Simplified Gibbs free energy minimisation.

        Iteratively adjusts mass fractions to minimise the total Gibbs
        free energy G = H - T*S using a gradient-descent-like approach.

        Parameters
        ----------
        T : float
            Temperature (K).
        initial_Y : sequence of float
            Initial mass fractions.
        thermo_models : sequence of JanafMultiThermoEnhanced8
            One thermo model per species.

        Returns
        -------
        list of float
            Equilibrium mass fractions.
        """
        Y = list(initial_Y)
        n = len(Y)

        for _ in range(self._gibbs_max_iter):
            # Compute species Gibbs contributions
            g = []
            for i, model in enumerate(thermo_models):
                h_i = float(model.Cp(T).item()) if hasattr(model.Cp(T), "item") else float(model.Cp(T))
                s_i = model.S_ref_at_T(T)
                g.append(h_i - T * s_i)

            # Gradient descent step
            g_avg = sum(Y[i] * g[i] for i in range(n))
            delta = 0.01
            max_change = 0.0
            for i in range(n):
                correction = -delta * (g[i] - g_avg)
                Y[i] = max(1e-10, Y[i] + correction)
                max_change = max(max_change, abs(correction))

            # Normalise
            total = sum(Y)
            if total > 0:
                Y = [y / total for y in Y]

            if max_change < self._gibbs_tol:
                break

        return Y

    def __repr__(self) -> str:
        total_L = self.total_latent_heat()
        has_table = "table" if self._S_ref_table is not None else "const"
        return (
            f"JanafMultiThermoEnhanced8(R={self._R}, n_phases={len(self._phases)}, "
            f"Hf={self._Hf}, blend_width={self._blend_width}, "
            f"S_ref_mode={has_table}, L_total={total_L:.0f}, "
            f"gibbs_max_iter={self._gibbs_max_iter})"
        )
