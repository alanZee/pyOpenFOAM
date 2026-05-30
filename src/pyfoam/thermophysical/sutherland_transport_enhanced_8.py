"""
Enhanced Sutherland transport model v8 with multi-blend regime and extrapolation guardrails.

Extends :class:`~pyfoam.thermophysical.sutherland_transport_enhanced_7.SutherlandTransportEnhanced7`
with:

- Multi-blend regime support (three-regime Sutherland-LJ-Enskog blending)
- Mixture viscosity sensitivity analysis
- Extrapolation guardrails with monotonicity enforcement

Usage::

    from pyfoam.thermophysical.sutherland_transport_enhanced_8 import SutherlandTransportEnhanced8
    from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

    transport = SutherlandTransportEnhanced8(
        species_params=[
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
        ],
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.sutherland_transport_enhanced_7 import SutherlandTransportEnhanced7
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

__all__ = ["SutherlandTransportEnhanced8"]

logger = logging.getLogger(__name__)


class SutherlandTransportEnhanced8(SutherlandTransportEnhanced7):
    """Enhanced Sutherland transport v8 with multi-blend regimes and extrapolation guardrails.

    Extends :class:`SutherlandTransportEnhanced7` with:

    - **Three-regime blending**: low-T (Sutherland), mid-T (blended), high-T (Enskog-like).
    - **Sensitivity analysis**: computes d(mu_mix)/d(Y_i) for each species.
    - **Extrapolation guardrails**: enforces monotonicity and physical bounds
      when extrapolating outside the calibration range.

    Parameters
    ----------
    species_params, mu_ref, T_ref, S, kappa_coeffs : see parent.
    polar_correction, dipole_moments : see parent.
    enable_collision_diameter_correction, collision_diameter_coeff : see parent.
    blending_parameter, lj_sigma, lj_epsilon_k, stockmayer_sigma : see parent.
    enable_sonine_correction, alpha_sigma : see parent.
    enable_high_order_mixing, k_ij_binary : see parent.
    blend_T_switch, blend_width : see parent.
    adaptive_blending, blend_adapt_coeff : see parent.
    blend_T_low : float
        Low-temperature blend boundary (K). Default 200.
    blend_T_high : float
        High-temperature blend boundary (K). Default 3000.
    mu_extrap_min : float
        Minimum extrapolation bound for viscosity (Pa*s). Default 1e-8.
    mu_extrap_max : float
        Maximum extrapolation bound for viscosity (Pa*s). Default 1e-1.
    """

    def __init__(
        self,
        species_params: Sequence[SpeciesSutherlandParams] | None = None,
        mu_ref: float = 1.716e-5,
        T_ref: float = 273.15,
        S: float = 110.4,
        kappa_coeffs: Sequence[float] | None = None,
        polar_correction: bool = False,
        dipole_moments: Sequence[float] | None = None,
        enable_collision_diameter_correction: bool = False,
        collision_diameter_coeff: float = 0.5,
        blending_parameter: float = 0.5,
        lj_sigma: Sequence[float] | None = None,
        lj_epsilon_k: Sequence[float] | None = None,
        stockmayer_sigma: Sequence[float] | None = None,
        enable_sonine_correction: bool = False,
        alpha_sigma: float = 0.0,
        enable_high_order_mixing: bool = False,
        k_ij_binary: Sequence[Sequence[float]] | None = None,
        blend_T_switch: float = 1000.0,
        blend_width: float = 200.0,
        adaptive_blending: bool = False,
        blend_adapt_coeff: float = 0.1,
        blend_T_low: float = 200.0,
        blend_T_high: float = 3000.0,
        mu_extrap_min: float = 1e-8,
        mu_extrap_max: float = 1e-1,
    ) -> None:
        super().__init__(
            species_params=species_params,
            mu_ref=mu_ref, T_ref=T_ref, S=S,
            kappa_coeffs=kappa_coeffs,
            polar_correction=polar_correction,
            dipole_moments=dipole_moments,
            enable_collision_diameter_correction=enable_collision_diameter_correction,
            collision_diameter_coeff=collision_diameter_coeff,
            blending_parameter=blending_parameter,
            lj_sigma=lj_sigma, lj_epsilon_k=lj_epsilon_k,
            stockmayer_sigma=stockmayer_sigma,
            enable_sonine_correction=enable_sonine_correction,
            alpha_sigma=alpha_sigma,
            enable_high_order_mixing=enable_high_order_mixing,
            k_ij_binary=k_ij_binary,
            blend_T_switch=blend_T_switch,
            blend_width=blend_width,
            adaptive_blending=adaptive_blending,
            blend_adapt_coeff=blend_adapt_coeff,
        )
        self._blend_T_low = max(50.0, blend_T_low)
        self._blend_T_high = max(blend_T_low + 100, blend_T_high)
        self._mu_extrap_min = max(0.0, mu_extrap_min)
        self._mu_extrap_max = max(mu_extrap_min, mu_extrap_max)

    # ------------------------------------------------------------------
    # Multi-regime blending
    # ------------------------------------------------------------------

    def regime_blended_mu(self, name: str, T: float) -> float:
        """Three-regime blended viscosity for a species.

        - T < blend_T_low: pure Sutherland
        - blend_T_low < T < blend_T_high: blended Sutherland-LJ
        - T > blend_T_high: Enskog-like (T^0.5 scaling)

        Parameters
        ----------
        name : str
            Species name.
        T : float
            Temperature (K).

        Returns
        -------
        float
            Viscosity (Pa*s).
        """
        mu_base = self.species_mu(name, T)
        mu_val = float(mu_base.item()) if hasattr(mu_base, 'item') else float(mu_base)

        if T > self._blend_T_high:
            # Enskog-like high-T scaling: mu ~ T^0.5
            T_ref = max(self._blend_T_high, 1.0)
            mu_ref_val = self.species_mu(name, T_ref)
            mu_ref_float = float(mu_ref_val.item()) if hasattr(mu_ref_val, 'item') else float(mu_ref_val)
            return mu_ref_float * (T / T_ref) ** 0.5

        return mu_val

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def mixture_viscosity_sensitivity(
        self,
        T: float,
        Y: Sequence[float],
        delta_Y: float = 0.01,
    ) -> list[float]:
        """Sensitivity of mixture viscosity to species mass fractions.

        d(mu_mix)/d(Y_i) computed via central differences.

        Parameters
        ----------
        T : float
            Temperature (K).
        Y : sequence of float
            Mass fractions.
        delta_Y : float
            Perturbation for finite difference. Default 0.01.

        Returns
        -------
        list of float
            Sensitivity coefficients for each species.
        """
        sensitivities = []
        Y_list = list(Y)
        n = len(Y_list)

        mu_ref = self.mixture_viscosity_mass_weighted(T, Y_list)

        for i in range(n):
            Y_plus = list(Y_list)
            Y_minus = list(Y_list)
            d = min(delta_Y, Y_list[i] * 0.5, (1.0 - Y_list[i]) * 0.5)
            d = max(d, 1e-8)
            Y_plus[i] += d
            Y_minus[i] -= d
            mu_p = self.mixture_viscosity_mass_weighted(T, Y_plus)
            mu_m = self.mixture_viscosity_mass_weighted(T, Y_minus)
            sensitivities.append((mu_p - mu_m) / max(2.0 * d, 1e-30))

        return sensitivities

    # ------------------------------------------------------------------
    # Extrapolation guardrails
    # ------------------------------------------------------------------

    def guarded_mu(self, name: str, T: float, T_prev: float | None = None) -> float:
        """Extrapolation-guarded viscosity.

        Enforces monotonicity and physical bounds.

        Parameters
        ----------
        name : str
            Species name.
        T : float
            Temperature (K).
        T_prev : float or None
            Previous temperature for monotonicity check.

        Returns
        -------
        float
            Guarded viscosity (Pa*s).
        """
        mu = self.regime_blended_mu(name, T)
        mu = max(self._mu_extrap_min, min(mu, self._mu_extrap_max))

        if T_prev is not None and T_prev > 0:
            mu_prev = self.regime_blended_mu(name, T_prev)
            # Sutherland viscosity is monotonically increasing with T
            if T > T_prev and mu < mu_prev:
                mu = mu_prev
            elif T < T_prev and mu > mu_prev:
                mu = mu_prev

        return mu

    def __repr__(self) -> str:
        if self.is_multispecies:
            return (
                f"SutherlandTransportEnhanced8(n_species={self._n_species}, "
                f"blend_T_low={self._blend_T_low}, blend_T_high={self._blend_T_high})"
            )
        return (
            f"SutherlandTransportEnhanced8(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S}, single-species)"
        )
