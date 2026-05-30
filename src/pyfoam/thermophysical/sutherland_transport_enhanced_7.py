"""
Enhanced Sutherland transport model v7 with adaptive blending and composition-aware mixing.

Extends :class:`~pyfoam.thermophysical.sutherland_transport_enhanced_6.SutherlandTransportEnhanced6`
with:

- Adaptive blending width based on temperature gradient
- Composition-aware mixing with mass-fraction weighting
- Viscosity ratio diagnostics for mixture quality assessment

Usage::

    from pyfoam.thermophysical.sutherland_transport_enhanced_7 import SutherlandTransportEnhanced7
    from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

    transport = SutherlandTransportEnhanced7(
        species_params=[
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
        ],
        enable_high_order_mixing=True,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.sutherland_transport_enhanced_6 import SutherlandTransportEnhanced6
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

__all__ = ["SutherlandTransportEnhanced7"]

logger = logging.getLogger(__name__)


class SutherlandTransportEnhanced7(SutherlandTransportEnhanced6):
    """Enhanced Sutherland transport v7 with adaptive blending and composition mixing.

    Extends :class:`SutherlandTransportEnhanced6` with:

    - **Adaptive blending**: adjusts the Sutherland-LJ blend width based on
      the local temperature gradient for smoother transitions.
    - **Composition-aware mixing**: mass-fraction-weighted mixture viscosity
      for multi-species systems instead of mole-fraction only.
    - **Viscosity ratio diagnostics**: computes the ratio of each species
      viscosity to the mixture mean for diagnostic purposes.

    Parameters
    ----------
    species_params, mu_ref, T_ref, S, kappa_coeffs : see parent.
    polar_correction, dipole_moments : see parent.
    enable_collision_diameter_correction, collision_diameter_coeff : see parent.
    blending_parameter, lj_sigma, lj_epsilon_k, stockmayer_sigma : see parent.
    enable_sonine_correction, alpha_sigma : see parent.
    enable_high_order_mixing, k_ij_binary : see parent.
    blend_T_switch, blend_width : see parent.
    adaptive_blending : bool
        Enable adaptive blend width based on local gradient. Default False.
    blend_adapt_coeff : float
        Adaptation coefficient for blend width. Default 0.1.
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
        )
        self._adaptive_blending = adaptive_blending
        self._blend_adapt_coeff = max(0.01, min(blend_adapt_coeff, 1.0))

    @property
    def adaptive_blending_enabled(self) -> bool:
        """Whether adaptive blending is active."""
        return self._adaptive_blending

    # ------------------------------------------------------------------
    # Adaptive blending
    # ------------------------------------------------------------------

    def adaptive_blend_width(self, T: float, dT: float = 0.0) -> float:
        """Compute adaptive blend width based on temperature and gradient.

        Parameters
        ----------
        T : float
            Temperature (K).
        dT : float
            Temperature gradient magnitude (K/m). Default 0.

        Returns
        -------
        float
            Adapted blend width (K).
        """
        if not self._adaptive_blending:
            return self._blend_width

        # Widen blending for large gradients
        base = self._blend_width
        adapt = base * (1.0 + self._blend_adapt_coeff * abs(dT) / max(T, 1.0))
        return min(adapt, base * 5.0)

    # ------------------------------------------------------------------
    # Composition-aware mixing
    # ------------------------------------------------------------------

    def mixture_viscosity_mass_weighted(
        self,
        T: float,
        Y: Sequence[float],
    ) -> float:
        """Mass-fraction-weighted mixture viscosity.

        mu_mix = sum_i Y_i * mu_i(T)

        Parameters
        ----------
        T : float
            Temperature (K).
        Y : sequence of float
            Mass fractions.

        Returns
        -------
        float
            Mixture viscosity (Pa*s).
        """
        if not self.is_multispecies:
            return self._mu_ref

        mu_mix = 0.0
        for i, y_i in enumerate(Y):
            name = self._species_names[i] if i < len(self._species_names) else str(i)
            mu_i = float(self.species_mu(name, T).item()) if hasattr(self.species_mu(name, T), 'item') else float(self.species_mu(name, T))
            mu_mix += y_i * mu_i
        return max(mu_mix, 1e-30)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def viscosity_ratios(self, T: float) -> list[float]:
        """Compute species-to-mixture viscosity ratios.

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        list of float
            Ratios mu_i / mu_mix for each species.
        """
        if not self.is_multispecies:
            return [1.0]

        mu_values = []
        for i in range(self._n_species):
            name = self._species_names[i] if i < len(self._species_names) else str(i)
            mu_val = self.species_mu(name, T)
            mu_values.append(float(mu_val.item()) if hasattr(mu_val, 'item') else float(mu_val))
        mu_mean = sum(mu_values) / max(len(mu_values), 1)
        return [m / max(mu_mean, 1e-30) for m in mu_values]

    def __repr__(self) -> str:
        if self.is_multispecies:
            adaptive = ", adaptive" if self._adaptive_blending else ""
            return (
                f"SutherlandTransportEnhanced7(n_species={self._n_species}, "
                f"blend_T_switch={self._T_ref}{adaptive})"
            )
        return (
            f"SutherlandTransportEnhanced7(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S}, single-species)"
        )
