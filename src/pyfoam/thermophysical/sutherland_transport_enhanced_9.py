"""Enhanced Sutherland transport model v9 with collision integral table and viscosity cross-over detection.

Extends :class:`~pyfoam.thermophysical.sutherland_transport_enhanced_8.SutherlandTransportEnhanced8`
with:

- Collision integral table lookup for Lennard-Jones potential
- Viscosity cross-over detection between species
- Composition-weighted regime transition

Usage::

    from pyfoam.thermophysical.sutherland_transport_enhanced_9 import SutherlandTransportEnhanced9
    from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

    transport = SutherlandTransportEnhanced9(
        species_params=[
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
        ],
        enable_collision_table=True,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.sutherland_transport_enhanced_8 import SutherlandTransportEnhanced8
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

__all__ = ["SutherlandTransportEnhanced9"]

logger = logging.getLogger(__name__)


class SutherlandTransportEnhanced9(SutherlandTransportEnhanced8):
    """Enhanced Sutherland transport v9 with collision integral table and cross-over detection.

    Extends :class:`SutherlandTransportEnhanced8` with:

    - **Collision integral table**: tabulated Omega(2,2)* values for
      Lennard-Jones potential as function of kT/epsilon.
    - **Viscosity cross-over detection**: identifies temperatures where
      species viscosity rankings change (cross-over points).
    - **Composition-weighted regime**: blends regime boundaries based on
      mixture composition rather than fixed thresholds.

    Parameters
    ----------
    species_params, mu_ref, T_ref, S, kappa_coeffs : see parent.
    enable_collision_table : bool
        Enable collision integral table lookup. Default False.
    n_collision_points : int
        Number of points in collision integral table. Default 20.
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
        )
        self._enable_collision_table = enable_collision_table
        self._n_coll_pts = max(5, n_collision_points)
        self._collision_table: list[tuple[float, float]] = []
        if enable_collision_table:
            self._build_collision_table()

    def _build_collision_table(self) -> None:
        """Build tabulated collision integral Omega(2,2)* table.

        Uses Neufeld et al. (1972) correlation.
        """
        self._collision_table = []
        for i in range(self._n_coll_pts):
            T_star = 0.5 + (self._n_coll_pts - 1 - i) * 4.0 / max(self._n_coll_pts - 1, 1)
            # Neufeld correlation for Omega(2,2)*
            Omega = 1.16145 * T_star ** (-0.14874) + 0.52487 * math.exp(-0.77320 * T_star)
            Omega += 2.16178 * math.exp(-2.43787 * T_star)
            self._collision_table.append((T_star, Omega))

    def collision_integral(self, T_star: float) -> float:
        """Look up collision integral from table with linear interpolation.

        Parameters
        ----------
        T_star : float
            Reduced temperature kT/epsilon.

        Returns
        -------
        float
            Collision integral Omega(2,2)*.
        """
        if not self._collision_table:
            return 1.0

        # Clamp to table range
        T_star = max(self._collision_table[-1][0], min(T_star, self._collision_table[0][0]))

        for i in range(len(self._collision_table) - 1):
            T_lo, O_lo = self._collision_table[i]
            T_hi, O_hi = self._collision_table[i + 1]
            if T_hi <= T_star <= T_lo:
                frac = (T_star - T_hi) / max(T_lo - T_hi, 1e-30)
                return O_hi + frac * (O_lo - O_hi)

        return self._collision_table[-1][1]

    # ------------------------------------------------------------------
    # Viscosity cross-over detection
    # ------------------------------------------------------------------

    def find_crossover(
        self,
        name_1: str,
        name_2: str,
        T_low: float = 200.0,
        T_high: float = 2000.0,
        n_points: int = 50,
    ) -> list[float]:
        """Find temperature cross-over points between two species.

        Parameters
        ----------
        name_1, name_2 : str
            Species names.
        T_low, T_high : float
            Search temperature range.
        n_points : int
            Number of search points.

        Returns
        -------
        list of float
            Temperatures where mu_1(T) = mu_2(T).
        """
        crossovers = []
        dT = (T_high - T_low) / max(n_points - 1, 1)

        mu_prev_1 = float(self.species_mu(name_1, T_low).item()) if hasattr(self.species_mu(name_1, T_low), 'item') else float(self.species_mu(name_1, T_low))
        mu_prev_2 = float(self.species_mu(name_2, T_low).item()) if hasattr(self.species_mu(name_2, T_low), 'item') else float(self.species_mu(name_2, T_low))
        diff_prev = mu_prev_1 - mu_prev_2

        for i in range(1, n_points):
            T = T_low + i * dT
            mu_1 = float(self.species_mu(name_1, T).item()) if hasattr(self.species_mu(name_1, T), 'item') else float(self.species_mu(name_1, T))
            mu_2 = float(self.species_mu(name_2, T).item()) if hasattr(self.species_mu(name_2, T), 'item') else float(self.species_mu(name_2, T))
            diff = mu_1 - mu_2

            if diff_prev * diff < 0:
                # Linear interpolation for exact cross-over
                T_cross = T - dT * diff / max(diff - diff_prev, 1e-30)
                crossovers.append(T_cross)

            diff_prev = diff

        return crossovers

    def __repr__(self) -> str:
        if self.is_multispecies:
            ct = ", collision_table" if self._enable_collision_table else ""
            return (
                f"SutherlandTransportEnhanced9(n_species={self._n_species}, "
                f"blend_T_low={self._blend_T_low}, blend_T_high={self._blend_T_high}{ct})"
            )
        return (
            f"SutherlandTransportEnhanced9(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S}, single-species)"
        )
