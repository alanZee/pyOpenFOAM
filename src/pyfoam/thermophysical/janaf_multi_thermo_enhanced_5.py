"""
Enhanced JANAF thermodynamic model v5 — multi-phase with reaction enthalpy.

Extends :class:`~pyfoam.thermophysical.janaf_multi_thermo_enhanced_4.JanafMultiThermoEnhanced4`
with:

- Reaction enthalpy computation from species formation enthalpies
- Temperature-dependent entropy via JANAF Cp integration
- Gibbs free energy of reaction for equilibrium constant estimation

Usage::

    from pyfoam.thermophysical.janaf_multi_thermo_enhanced_5 import JanafMultiThermoEnhanced5
    from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

    phases = [
        JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
        JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
    ]
    thermo = JanafMultiThermoEnhanced5(R=461.5, phases=phases, blend_width=5.0)
    S = thermo.entropy(500.0)
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_4 import JanafMultiThermoEnhanced4
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase

__all__ = ["JanafMultiThermoEnhanced5"]

logger = logging.getLogger(__name__)


class JanafMultiThermoEnhanced5(JanafMultiThermoEnhanced4):
    """Enhanced multi-phase JANAF v5 with reaction enthalpy and entropy.

    Extends :class:`JanafMultiThermoEnhanced4` with:

    - **Reaction enthalpy**: Delta_H_rxn from formation enthalpies of
      products minus reactants at a given temperature.
    - **Entropy computation**: S(T) = S_ref + integral(Cp/T)dT via
      trapezoidal JANAF polynomial integration.
    - **Gibbs free energy of reaction**: Delta_G_rxn = Delta_H_rxn - T * Delta_S_rxn
      for equilibrium constant estimation: K_eq = exp(-Delta_G / (R*T)).

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
    ) -> None:
        super().__init__(
            R=R, phases=phases, Hf=Hf, blend_width=blend_width,
            steepness=steepness, beta_coeffs=beta_coeffs,
            latent_sensible_fraction=latent_sensible_fraction,
        )
        self._S_ref = S_ref
        self._T_ref_entropy = T_ref

    @property
    def S_ref(self) -> float:
        """Reference entropy (J/(kg*K))."""
        return self._S_ref

    @property
    def T_ref_entropy(self) -> float:
        """Reference temperature for entropy (K)."""
        return self._T_ref_entropy

    # ------------------------------------------------------------------
    # Entropy computation
    # ------------------------------------------------------------------

    def entropy(self, T: float, n_steps: int = 100) -> float:
        """Compute entropy at temperature T via Cp/T integration.

        S(T) = S_ref + integral(T_ref..T, Cp/T') dT'

        Uses trapezoidal rule over the JANAF polynomial Cp.

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
            return self._S_ref

        dt = (T - T0) / n_steps
        S = self._S_ref

        for i in range(n_steps):
            T_lo = T0 + i * dt
            T_hi = T0 + (i + 1) * dt
            T_mid = 0.5 * (T_lo + T_hi)
            cp_lo = float(self.Cp(T_lo).item()) if hasattr(self.Cp(T_lo), 'item') else float(self.Cp(T_lo))
            cp_hi = float(self.Cp(T_hi).item()) if hasattr(self.Cp(T_hi), 'item') else float(self.Cp(T_hi))
            cp_mid = float(self.Cp(T_mid).item()) if hasattr(self.Cp(T_mid), 'item') else float(self.Cp(T_mid))

            # Trapezoidal: (f(a) + 4*f(mid) + f(b)) / 6 * h  (Simpson's rule)
            integrand_lo = cp_lo / max(T_lo, 1e-10)
            integrand_hi = cp_hi / max(T_hi, 1e-10)
            integrand_mid = cp_mid / max(T_mid, 1e-10)
            S += abs(dt) * (integrand_lo + 4.0 * integrand_mid + integrand_hi) / 6.0

        return S

    # ------------------------------------------------------------------
    # Reaction enthalpy
    # ------------------------------------------------------------------

    def reaction_enthalpy(
        self,
        T: float,
        Hf_products: float,
        Hf_reactants: float,
    ) -> float:
        """Reaction enthalpy at temperature T.

        Delta_H_rxn(T) = H_products(T) - H_reactants(T)
                        = (Hf_products + Delta_H_sensible_products)
                        - (Hf_reactants + Delta_H_sensible_reactants)

        For simplicity this uses the global Hf and Cp to compute
        sensible enthalpy change from T_ref.

        Parameters
        ----------
        T : float
            Temperature (K).
        Hf_products : float
            Total formation enthalpy of products (J/kg).
        Hf_reactants : float
            Total formation enthalpy of reactants (J/kg).

        Returns
        -------
        float
            Reaction enthalpy (J/kg).
        """
        # Sensible enthalpy change from T_ref to T
        T0 = self._T_ref_entropy
        n_steps = 50
        dt = (T - T0) / max(n_steps, 1)
        H_sens = 0.0
        for i in range(n_steps):
            T_lo = T0 + i * dt
            T_hi = T0 + (i + 1) * dt
            T_mid = 0.5 * (T_lo + T_hi)
            cp_mid = float(self.Cp(T_mid).item()) if hasattr(self.Cp(T_mid), 'item') else float(self.Cp(T_mid))
            H_sens += cp_mid * abs(dt)

        return (Hf_products + H_sens) - (Hf_reactants + H_sens)

    # ------------------------------------------------------------------
    # Gibbs free energy of reaction
    # ------------------------------------------------------------------

    def gibbs_reaction(
        self,
        T: float,
        Hf_products: float,
        Hf_reactants: float,
        S_products: float,
        S_reactants: float,
    ) -> float:
        """Gibbs free energy of reaction.

        Delta_G = Delta_H - T * Delta_S

        Parameters
        ----------
        T : float
            Temperature (K).
        Hf_products, Hf_reactants : float
            Formation enthalpies (J/kg).
        S_products, S_reactants : float
            Entropies (J/(kg*K)).

        Returns
        -------
        float
            Gibbs free energy of reaction (J/kg).
        """
        dH = self.reaction_enthalpy(T, Hf_products, Hf_reactants)
        dS = S_products - S_reactants
        return dH - T * dS

    def equilibrium_constant(
        self,
        T: float,
        dG: float,
    ) -> float:
        """Equilibrium constant from Gibbs energy.

        K_eq = exp(-Delta_G / (R * T))

        Parameters
        ----------
        T : float
            Temperature (K).
        dG : float
            Gibbs free energy of reaction (J/kg).

        Returns
        -------
        float
            Equilibrium constant.
        """
        exponent = -dG / (self._R * max(T, 1.0))
        exponent = max(min(exponent, 500.0), -500.0)
        import math
        return math.exp(exponent)

    def __repr__(self) -> str:
        total_L = self.total_latent_heat()
        return (
            f"JanafMultiThermoEnhanced5(R={self._R}, n_phases={len(self._phases)}, "
            f"Hf={self._Hf}, blend_width={self._blend_width}, "
            f"S_ref={self._S_ref}, L_total={total_L:.0f})"
        )
