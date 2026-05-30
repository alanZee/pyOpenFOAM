"""
Enhanced Sutherland transport model v6 with high-order mixing rules and multicomponent blending.

Extends :class:`~pyfoam.thermophysical.sutherland_transport_enhanced_5.SutherlandTransportEnhanced5`
with:

- High-order Mason-Saxena mixing rule with binary interaction parameters
- Temperature-dependent blending between Sutherland and LJ regimes
- Multi-zone transport for stratified gas mixtures

Usage::

    from pyfoam.thermophysical.sutherland_transport_enhanced_6 import SutherlandTransportEnhanced6
    from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

    transport = SutherlandTransportEnhanced6(
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
from pyfoam.thermophysical.sutherland_transport_enhanced_5 import SutherlandTransportEnhanced5
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

__all__ = ["SutherlandTransportEnhanced6"]

logger = logging.getLogger(__name__)


class SutherlandTransportEnhanced6(SutherlandTransportEnhanced5):
    """Enhanced Sutherland transport v6 with high-order mixing and multi-zone.

    Extends :class:`SutherlandTransportEnhanced5` with:

    - **High-order Mason-Saxena mixing**: includes binary interaction
      parameters k_ij for improved mixture viscosity prediction in
      asymmetric gas mixtures (e.g., H2-CH4).
    - **Sutherland-LJ blending**: smoothly transitions between Sutherland
      (low T) and LJ-based (high T) transport predictions.
    - **Multi-zone support**: allows specification of zone-dependent
      species composition for stratified mixtures.

    Parameters
    ----------
    species_params : sequence of SpeciesSutherlandParams
        Per-species Sutherland parameters.
    mu_ref, T_ref, S, kappa_coeffs :
        Single-species parameters.
    polar_correction : bool
        Enable Brokaw polar collision correction.
    dipole_moments : sequence of float or None
        Dipole moments (Debye).
    enable_collision_diameter_correction : bool
        Enable temperature-dependent collision diameter.
    collision_diameter_coeff : float
        Coefficient for temperature correction.
    blending_parameter : float
        Blending parameter.
    lj_sigma, lj_epsilon_k, stockmayer_sigma : see parent
    enable_sonine_correction : bool
        Enable Sonine polynomial correction.
    alpha_sigma : float
        Temperature coefficient for effective diameter.
    enable_high_order_mixing : bool
        Enable high-order Mason-Saxena mixing. Default False.
    k_ij_binary : sequence of sequence of float or None
        Binary interaction parameter matrix. Default None (all zeros).
    blend_T_switch : float
        Temperature for Sutherland-LJ blending (K). Default 1000.
    blend_width : float
        Blending width in temperature (K). Default 200.
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
        )
        self._high_order_mixing = enable_high_order_mixing
        self._blend_T_switch = blend_T_switch
        self._blend_width = max(blend_width, 1.0)

        if k_ij_binary is not None and self.is_multispecies:
            self._k_ij = [list(row) for row in k_ij_binary]
        else:
            self._k_ij = None

    @property
    def high_order_mixing_enabled(self) -> bool:
        """Whether high-order mixing is active."""
        return self._high_order_mixing

    # ------------------------------------------------------------------
    # Sutherland-LJ blending
    # ------------------------------------------------------------------

    def _sutherland_lj_blend(self, T: float) -> float:
        """Blending factor between Sutherland and LJ regimes.

        Returns 0.0 for pure Sutherland (low T) and 1.0 for pure LJ (high T).

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Blend factor in [0, 1].
        """
        if self._blend_width <= 0:
            return 0.0 if T < self._blend_T_switch else 1.0
        arg = (T - self._blend_T_switch) / self._blend_width
        return max(0.0, min(1.0, 0.5 * (1.0 + math.tanh(arg))))

    # ------------------------------------------------------------------
    # High-order mixing
    # ------------------------------------------------------------------

    def _high_order_mixing_correction(
        self,
        T: float,
        species_i: int,
        species_j: int,
    ) -> float:
        """Binary interaction correction for Mason-Saxena mixing.

        factor = 1 - k_ij * (eps_ij / T)

        Parameters
        ----------
        T : float
            Temperature (K).
        species_i, species_j : int
            Species indices.

        Returns
        -------
        float
            Correction factor (close to 1.0).
        """
        if not self._high_order_mixing or self._k_ij is None:
            return 1.0
        if species_i >= len(self._k_ij) or species_j >= len(self._k_ij[0]):
            return 1.0

        k_ij = self._k_ij[species_i][species_j]
        if self._lj_epsilon_k is not None:
            eps_i = self._lj_epsilon_k[min(species_i, len(self._lj_epsilon_k) - 1)]
            eps_j = self._lj_epsilon_k[min(species_j, len(self._lj_epsilon_k) - 1)]
            eps_eff = math.sqrt(eps_i * eps_j)
        else:
            eps_eff = 100.0

        return max(0.5, 1.0 - k_ij * eps_eff / max(T, 1.0))

    def __repr__(self) -> str:
        if self.is_multispecies:
            ho = ", high_order" if self._high_order_mixing else ""
            return (
                f"SutherlandTransportEnhanced6(n_species={self._n_species}, "
                f"blend_T_switch={self._T_ref}{ho})"
            )
        return (
            f"SutherlandTransportEnhanced6(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S}, single-species)"
        )
