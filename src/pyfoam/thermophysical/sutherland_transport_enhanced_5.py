"""
Enhanced Sutherland transport model v5 with Stockmayer potential and Sonine polynomials.

Extends :class:`~pyfoam.thermophysical.sutherland_transport_enhanced_4.SutherlandTransportEnhanced4`
with:

- Stockmayer potential collision integrals for polar gases
- Higher-order Sonine polynomial approximations for Chapman-Enskog
- Temperature-dependent effective diameter from intermolecular potential

Usage::

    from pyfoam.thermophysical.sutherland_transport_enhanced_5 import SutherlandTransportEnhanced5
    from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

    transport = SutherlandTransportEnhanced5(
        species_params=[
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            SpeciesSutherlandParams(name="H2O", mu_ref=1.0e-5, T_ref=373.15, S=350.0, Mw=18.015),
        ],
        lj_sigma=[3.798, 2.641],
        lj_epsilon_k=[71.4, 809.1],
        dipole_moments=[0.0, 1.85],
        stockmayer_sigma=[3.798, 2.641],
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.sutherland_transport_enhanced_4 import SutherlandTransportEnhanced4
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

__all__ = ["SutherlandTransportEnhanced5"]

logger = logging.getLogger(__name__)


class SutherlandTransportEnhanced5(SutherlandTransportEnhanced4):
    """Enhanced Sutherland transport v5 with Stockmayer and Sonine.

    Extends :class:`SutherlandTransportEnhanced4` with:

    - **Stockmayer collision integrals**: for polar molecules with
      dipole moments, uses the Stockmayer potential which accounts for
      the orientation-dependent dipole-dipole interaction. More accurate
      than LJ for polar gases like H2O, NH3, SO2.
    - **Sonine polynomial expansion**: higher-order Chapman-Enskog
      approximation using [2/2] Padé approximant for transport
      properties, improving accuracy at intermediate T*.
    - **Temperature-dependent effective diameter**: sigma(T) = sigma_0
      * (1 + alpha_sigma * (T - T_ref) / T_ref) for better temperature
      extrapolation of collision cross-section.

    Parameters
    ----------
    species_params : sequence of SpeciesSutherlandParams
        Per-species Sutherland parameters.
    mu_ref, T_ref, S, kappa_coeffs :
        Single-species parameters.
    polar_correction : bool
        Enable Brokaw polar collision correction.
    dipole_moments : sequence of float or None
        Dipole moments (Debye) for each species.
    enable_collision_diameter_correction : bool
        Enable temperature-dependent collision diameter.
    collision_diameter_coeff : float
        Coefficient for temperature correction.
    blending_parameter : float
        Blending parameter.
    lj_sigma : sequence of float or None
        Lennard-Jones collision diameters (Angstrom) per species.
    lj_epsilon_k : sequence of float or None
        Lennard-Jones well depths (K) per species.
    stockmayer_sigma : sequence of float or None
        Stockmayer hard-sphere diameters (Angstrom) per species.
    enable_sonine_correction : bool
        Enable Sonine polynomial correction. Default False.
    alpha_sigma : float
        Temperature coefficient for effective diameter. Default 0.0.
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
            lj_sigma=lj_sigma,
            lj_epsilon_k=lj_epsilon_k,
        )
        self._stockmayer_sigma = list(stockmayer_sigma) if stockmayer_sigma is not None else None
        self._sonine_correction = enable_sonine_correction
        self._alpha_sigma = alpha_sigma

    @property
    def has_stockmayer_sigma(self) -> bool:
        """Whether Stockmayer hard-sphere diameters are available."""
        return self._stockmayer_sigma is not None

    @property
    def sonine_correction_enabled(self) -> bool:
        """Whether Sonine polynomial correction is active."""
        return self._sonine_correction

    # ------------------------------------------------------------------
    # Stockmayer collision integral
    # ------------------------------------------------------------------

    def stockmayer_collision_integral(
        self,
        T: float,
        species_i: int,
        species_j: int,
    ) -> float:
        """Stockmayer potential collision integral.

        For polar gas pairs, uses the Stockmayer (12-6-3) potential:
        Omega_st = Omega_LJ * (1 + delta^2 / (8 * T*))

        where delta = mu_dipole^2 / (eps * sigma^3).

        Parameters
        ----------
        T : float
            Temperature (K).
        species_i, species_j : int
            Species indices.

        Returns
        -------
        float
            Collision integral (dimensionless).
        """
        if (self._dipole_moments is None or self._lj_epsilon_k is None
                or self._stockmayer_sigma is None):
            return 1.0

        eps_i = self._lj_epsilon_k[species_i]
        eps_j = self._lj_epsilon_k[species_j]
        eps_eff = math.sqrt(eps_i * eps_j)

        sigma_i = self._stockmayer_sigma[species_i]
        sigma_j = self._stockmayer_sigma[species_j]
        sigma_eff = 0.5 * (sigma_i + sigma_j)

        T_star = T / max(eps_eff, 1e-10)
        T_star = max(T_star, 0.3)

        # LJ baseline
        Omega_LJ = 1.147 / (T_star ** 0.145)

        # Reduced dipole moment
        mu_i = self._dipole_moments[species_i]
        mu_j = self._dipole_moments[species_j]
        mu_eff = math.sqrt(abs(mu_i * mu_j))
        # delta = mu^2 / (eps * sigma^3)  (in Debye^2 / (K * A^3))
        # Convert: 1 Debye^2 = 1e-49 C^2*m^2; for dimensionless: scale factor
        delta_sq = (mu_eff * 3.162e-6) ** 2 / max(eps_eff * sigma_eff ** 3, 1e-30)

        correction = 1.0 + delta_sq / (8.0 * max(T_star, 0.1))
        return Omega_LJ * correction

    # ------------------------------------------------------------------
    # Sonine polynomial correction
    # ------------------------------------------------------------------

    def _sonine_correction_factor(self, T_star: float) -> float:
        """[2/2] Padé approximant for Sonine polynomial correction.

        The first-order Chapman-Enskog result is multiplied by this factor
        for improved accuracy at intermediate T*.

        factor = (1 + a1/T* + a2/T*^2) / (1 + b1/T* + b2/T*^2)

        Parameters
        ----------
        T_star : float
            Reduced temperature.

        Returns
        -------
        float
            Sonine correction factor (close to 1.0).
        """
        if not self._sonine_correction:
            return 1.0

        T_s = max(T_star, 0.3)
        a1, a2 = 0.12, 0.02
        b1, b2 = 0.15, 0.03

        num = 1.0 + a1 / T_s + a2 / T_s ** 2
        den = 1.0 + b1 / T_s + b2 / T_s ** 2
        return num / max(den, 0.1)

    # ------------------------------------------------------------------
    # Temperature-dependent effective diameter
    # ------------------------------------------------------------------

    def effective_sigma_T(self, T: float, species_i: int, species_j: int) -> float:
        """Temperature-dependent effective collision diameter.

        sigma(T) = sigma_eff * (1 + alpha * (T - T_ref) / T_ref)

        Parameters
        ----------
        T : float
            Temperature (K).
        species_i, species_j : int
            Species indices.

        Returns
        -------
        float
            Effective collision diameter (Angstrom).
        """
        sigma_base = self.effective_sigma(species_i, species_j)
        dT_rel = (T - self._T_ref) / max(self._T_ref, 1.0)
        return sigma_base * (1.0 + self._alpha_sigma * dT_rel)

    def __repr__(self) -> str:
        if self.is_multispecies:
            polar = ", polar" if self._polar_correction else ""
            lj = ", LJ" if self.has_lj_params else ""
            stock = ", Stockmayer" if self.has_stockmayer_sigma else ""
            sonine = ", Sonine" if self._sonine_correction else ""
            return (
                f"SutherlandTransportEnhanced5(n_species={self._n_species}, "
                f"species={self._species_names}{polar}{lj}{stock}{sonine})"
            )
        return (
            f"SutherlandTransportEnhanced5(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S}, single-species)"
        )
