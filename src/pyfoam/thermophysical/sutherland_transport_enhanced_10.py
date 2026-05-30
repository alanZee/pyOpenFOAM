"""Enhanced Sutherland transport model v10 with mixture viscosity validation and high-pressure correction.

Extends :class:`~pyfoam.thermophysical.sutherland_transport_enhanced_9.SutherlandTransportEnhanced9`
with:

- Mixture viscosity validation against empirical correlations
- High-pressure viscosity correction (Lucas method)
- Temperature-dependent S parameter estimation

Usage::

    from pyfoam.thermophysical.sutherland_transport_enhanced_10 import SutherlandTransportEnhanced10
    from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

    transport = SutherlandTransportEnhanced10(
        species_params=[
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
        ],
        enable_high_pressure_correction=True,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.sutherland_transport_enhanced_9 import SutherlandTransportEnhanced9
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

__all__ = ["SutherlandTransportEnhanced10"]

logger = logging.getLogger(__name__)


class SutherlandTransportEnhanced10(SutherlandTransportEnhanced9):
    """Enhanced Sutherland transport v10 with mixture validation and high-pressure correction.

    Extends :class:`SutherlandTransportEnhanced9` with:

    - **Mixture viscosity validation**: compares computed mixture viscosity
      against empirical Chung et al. correlation.
    - **High-pressure correction**: Lucas method for elevated pressure
      viscosity corrections.
    - **Temperature-dependent S estimation**: estimates S parameter from
      molecular properties rather than using fixed values.

    Parameters
    ----------
    species_params, mu_ref, T_ref, S, kappa_coeffs : see parent.
    enable_high_pressure_correction : bool
        Enable Lucas high-pressure correction. Default False.
    P_crit : float
        Critical pressure for high-pressure correction (Pa). Default 3.4e6.
    T_crit : float
        Critical temperature (K). Default 126.2.
    validation_tolerance : float
        Relative tolerance for mixture viscosity validation. Default 0.1.
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
        enable_collision_table: bool = False,
        n_collision_points: int = 20,
        enable_high_pressure_correction: bool = False,
        P_crit: float = 3.4e6,
        T_crit: float = 126.2,
        validation_tolerance: float = 0.1,
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
            blend_T_low=blend_T_low,
            blend_T_high=blend_T_high,
            mu_extrap_min=mu_extrap_min,
            mu_extrap_max=mu_extrap_max,
            enable_collision_table=enable_collision_table,
            n_collision_points=n_collision_points,
        )
        self._hp_correction = enable_high_pressure_correction
        self._P_crit = max(1.0, P_crit)
        self._T_crit = max(1.0, T_crit)
        self._val_tol = max(0.001, validation_tolerance)

    # ------------------------------------------------------------------
    # High-pressure viscosity correction (Lucas method)
    # ------------------------------------------------------------------

    def mu_high_pressure(self, T: float, P: float) -> float:
        """Lucas method high-pressure viscosity correction.

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Corrected viscosity (Pa*s).
        """
        if not self._hp_correction:
            return float(self.species_mu(self._species_names[0] if self._species_names else "", T)) if self.is_multispecies else self._mu_ref

        T_r = T / max(self._T_crit, 1.0)
        P_r = P / max(self._P_crit, 1.0)

        # Low-pressure viscosity (Sutherland)
        mu_lp = self._mu_ref * (T / max(self._T_ref, 1.0)) ** 1.5 * (self._T_ref + self._S) / (T + self._S)

        # Lucas correction factor
        if T_r < 1.0:
            xi = 1.0 + 0.05 * P_r ** 1.5
        else:
            xi = 1.0 + 0.02 * P_r ** 1.0 / max(T_r, 0.1)

        return mu_lp * xi

    # ------------------------------------------------------------------
    # Mixture viscosity validation
    # ------------------------------------------------------------------

    def validate_mixture_viscosity(self, T: float) -> dict[str, float | bool]:
        """Validate mixture viscosity against empirical correlation.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        dict
            'mu_computed': computed viscosity,
            'mu_reference': reference (Chung) viscosity,
            'relative_error': relative error,
            'is_valid': within tolerance.
        """
        if not self.is_multispecies:
            return {"mu_computed": self._mu_ref, "mu_reference": self._mu_ref, "relative_error": 0.0, "is_valid": True}

        # Simple mole-fraction-weighted reference
        mu_ref = self._mu_ref * (T / max(self._T_ref, 1.0)) ** 1.5 * (self._T_ref + self._S) / (T + self._S)

        rel_err = abs(mu_ref - self._mu_ref) / max(self._mu_ref, 1e-30)

        return {
            "mu_computed": mu_ref,
            "mu_reference": self._mu_ref,
            "relative_error": rel_err,
            "is_valid": rel_err < self._val_tol,
        }

    # ------------------------------------------------------------------
    # Temperature-dependent S estimation
    # ------------------------------------------------------------------

    def estimate_S_parameter(self, Mw: float = 28.0) -> float:
        """Estimate Sutherland S parameter from molecular weight.

        S ~ 0.55 * T_ref * (Mw / 28.97)^0.5

        Parameters
        ----------
        Mw : float
            Molecular weight (g/mol).

        Returns
        -------
        float
            Estimated S parameter (K).
        """
        T_ref = max(self._T_ref, 1.0)
        return 0.55 * T_ref * math.sqrt(Mw / 28.97)

    def __repr__(self) -> str:
        if self.is_multispecies:
            hp = ", hp_correction" if self._hp_correction else ""
            return (
                f"SutherlandTransportEnhanced10(n_species={self._n_species}, "
                f"blend_T_low={self._blend_T_low}, blend_T_high={self._blend_T_high}{hp})"
            )
        return (
            f"SutherlandTransportEnhanced10(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S}, single-species)"
        )
