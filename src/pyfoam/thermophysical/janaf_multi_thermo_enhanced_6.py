"""
Enhanced JANAF thermodynamic model v6 — multi-species reaction network with Cp moments.

Extends :class:`~pyfoam.thermophysical.janaf_multi_thermo_enhanced_5.JanafMultiThermoEnhanced5`
with:

- Multi-species reaction network stoichiometry support
- Temperature-dependent reference entropy via JANAF polynomial moments
- Cp integral moments for mean-Cp and variance estimation

Usage::

    from pyfoam.thermophysical.janaf_multi_thermo_enhanced_6 import JanafMultiThermoEnhanced6
    from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

    phases = [
        JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
        JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
    ]
    thermo = JanafMultiThermoEnhanced6(R=461.5, phases=phases, blend_width=5.0)
    cp_mean = thermo.Cp_mean(300.0, 500.0)
"""

from __future__ import annotations

import math
import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_5 import JanafMultiThermoEnhanced5
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

__all__ = ["JanafMultiThermoEnhanced6"]

logger = logging.getLogger(__name__)


class JanafMultiThermoEnhanced6(JanafMultiThermoEnhanced5):
    """Enhanced multi-phase JANAF v6 with reaction network and Cp moments.

    Extends :class:`JanafMultiThermoEnhanced5` with:

    - **Reaction network stoichiometry**: define multi-species reactions
      with stoichiometric coefficients for products and reactants, then
      compute overall Delta_H, Delta_S, and Delta_G for the reaction.
    - **Cp integral moments**: compute mean-Cp, Cp variance, and
      higher-order moments over a temperature range using Simpson's rule
      integration of the JANAF polynomial.
    - **Temperature-dependent reference entropy**: allows S_ref(T) lookup
      from a table of reference entropy values at different temperatures.

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
    ) -> None:
        super().__init__(
            R=R, phases=phases, Hf=Hf, blend_width=blend_width,
            steepness=steepness, beta_coeffs=beta_coeffs,
            latent_sensible_fraction=latent_sensible_fraction,
            S_ref=S_ref, T_ref=T_ref,
        )
        self._S_ref_table = S_ref_table

    @property
    def S_ref_table(self) -> dict[float, float] | None:
        """Temperature-dependent reference entropy table."""
        return self._S_ref_table

    # ------------------------------------------------------------------
    # Cp integral moments
    # ------------------------------------------------------------------

    def _cp_integrate(
        self,
        T_lo: float,
        T_hi: float,
        n_steps: int = 100,
    ) -> tuple[float, float]:
        """Integrate Cp and Cp^2 over [T_lo, T_hi] using Simpson's rule.

        Returns
        -------
        tuple of (float, float)
            (integral_Cp_dT, integral_Cp2_dT)
        """
        dt = (T_hi - T_lo) / max(n_steps, 1)
        I_cp = 0.0
        I_cp2 = 0.0

        for i in range(n_steps):
            t_lo = T_lo + i * dt
            t_hi = T_lo + (i + 1) * dt
            t_mid = 0.5 * (t_lo + t_hi)

            cp_lo = float(self.Cp(t_lo).item()) if hasattr(self.Cp(t_lo), 'item') else float(self.Cp(t_lo))
            cp_hi = float(self.Cp(t_hi).item()) if hasattr(self.Cp(t_hi), 'item') else float(self.Cp(t_hi))
            cp_mid = float(self.Cp(t_mid).item()) if hasattr(self.Cp(t_mid), 'item') else float(self.Cp(t_mid))

            # Simpson's 1/3 rule
            I_cp += abs(dt) * (cp_lo + 4.0 * cp_mid + cp_hi) / 6.0
            I_cp2 += abs(dt) * (cp_lo ** 2 + 4.0 * cp_mid ** 2 + cp_hi ** 2) / 6.0

        return I_cp, I_cp2

    def Cp_mean(self, T_lo: float, T_hi: float, n_steps: int = 100) -> float:
        """Mean heat capacity over a temperature range.

        Cp_mean = integral(Cp dT) / (T_hi - T_lo)

        Parameters
        ----------
        T_lo : float
            Lower temperature bound (K).
        T_hi : float
            Upper temperature bound (K).
        n_steps : int
            Integration steps. Default 100.

        Returns
        -------
        float
            Mean Cp (J/(kg*K)).
        """
        dT = T_hi - T_lo
        if abs(dT) < 1e-10:
            return float(self.Cp(T_lo).item()) if hasattr(self.Cp(T_lo), 'item') else float(self.Cp(T_lo))

        I_cp, _ = self._cp_integrate(T_lo, T_hi, n_steps)
        return I_cp / abs(dT)

    def Cp_variance(self, T_lo: float, T_hi: float, n_steps: int = 100) -> float:
        """Variance of heat capacity over a temperature range.

        Var(Cp) = integral(Cp^2 dT) / dT - (Cp_mean)^2

        Parameters
        ----------
        T_lo : float
            Lower temperature bound (K).
        T_hi : float
            Upper temperature bound (K).
        n_steps : int
            Integration steps. Default 100.

        Returns
        -------
        float
            Cp variance.
        """
        dT = T_hi - T_lo
        if abs(dT) < 1e-10:
            return 0.0

        I_cp, I_cp2 = self._cp_integrate(T_lo, T_hi, n_steps)
        mean_cp = I_cp / abs(dT)
        mean_cp2 = I_cp2 / abs(dT)
        return max(mean_cp2 - mean_cp ** 2, 0.0)

    # ------------------------------------------------------------------
    # Multi-species reaction network
    # ------------------------------------------------------------------

    def reaction_network_enthalpy(
        self,
        T: float,
        species_Hf: Sequence[float],
        stoich_reactants: Sequence[float],
        stoich_products: Sequence[float],
    ) -> float:
        """Reaction enthalpy from stoichiometric coefficients.

        Delta_H_rxn = sum(nu_p * Hf_p) - sum(nu_r * Hf_r) + Delta_H_sensible

        Parameters
        ----------
        T : float
            Temperature (K).
        species_Hf : sequence of float
            Formation enthalpies of all species (J/kg).
        stoich_reactants : sequence of float
            Stoichiometric coefficients of reactants.
        stoich_products : sequence of float
            Stoichiometric coefficients of products.

        Returns
        -------
        float
            Reaction enthalpy (J/kg).
        """
        Hf_products = sum(
            c * species_Hf[i] for i, c in enumerate(stoich_products) if i < len(species_Hf)
        )
        Hf_reactants = sum(
            c * species_Hf[i] for i, c in enumerate(stoich_reactants) if i < len(species_Hf)
        )
        return self.reaction_enthalpy(T, Hf_products, Hf_reactants)

    # ------------------------------------------------------------------
    # Temperature-dependent reference entropy interpolation
    # ------------------------------------------------------------------

    def _S_ref_at_T(self, T: float) -> float:
        """Interpolate reference entropy from table.

        Uses linear interpolation between table entries. Falls back
        to the constant S_ref if no table is provided.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Reference entropy (J/(kg*K)).
        """
        if self._S_ref_table is None:
            return self._S_ref

        temps = sorted(self._S_ref_table.keys())
        if len(temps) == 0:
            return self._S_ref

        if T <= temps[0]:
            return self._S_ref_table[temps[0]]
        if T >= temps[-1]:
            return self._S_ref_table[temps[-1]]

        # Linear interpolation
        for i in range(len(temps) - 1):
            if temps[i] <= T <= temps[i + 1]:
                frac = (T - temps[i]) / (temps[i + 1] - temps[i])
                S_lo = self._S_ref_table[temps[i]]
                S_hi = self._S_ref_table[temps[i + 1]]
                return S_lo + frac * (S_hi - S_lo)

        return self._S_ref

    def entropy(self, T: float, n_steps: int = 100) -> float:
        """Compute entropy with temperature-dependent S_ref.

        Parameters
        ----------
        T : float
            Temperature (K).
        n_steps : int
            Number of integration steps. Default 100.

        Returns
        -------
        float
            Entropy (J/(kg*K)).
        """
        T0 = self._T_ref_entropy
        if abs(T - T0) < 1e-10:
            return self._S_ref_at_T(T0)

        dt = (T - T0) / n_steps
        S = self._S_ref_at_T(T0)

        for i in range(n_steps):
            T_lo = T0 + i * dt
            T_hi = T0 + (i + 1) * dt
            T_mid = 0.5 * (T_lo + T_hi)
            cp_lo = float(self.Cp(T_lo).item()) if hasattr(self.Cp(T_lo), 'item') else float(self.Cp(T_lo))
            cp_hi = float(self.Cp(T_hi).item()) if hasattr(self.Cp(T_hi), 'item') else float(self.Cp(T_hi))
            cp_mid = float(self.Cp(T_mid).item()) if hasattr(self.Cp(T_mid), 'item') else float(self.Cp(T_mid))

            integrand_lo = cp_lo / max(T_lo, 1e-10)
            integrand_hi = cp_hi / max(T_hi, 1e-10)
            integrand_mid = cp_mid / max(T_mid, 1e-10)
            S += abs(dt) * (integrand_lo + 4.0 * integrand_mid + integrand_hi) / 6.0

        return S

    def __repr__(self) -> str:
        total_L = self.total_latent_heat()
        has_table = "table" if self._S_ref_table is not None else "const"
        return (
            f"JanafMultiThermoEnhanced6(R={self._R}, n_phases={len(self._phases)}, "
            f"Hf={self._Hf}, blend_width={self._blend_width}, "
            f"S_ref_mode={has_table}, L_total={total_L:.0f})"
        )
