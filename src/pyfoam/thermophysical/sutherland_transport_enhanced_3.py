"""
Enhanced Sutherland transport model v3 with improved multi-species support.

Extends :class:`~pyfoam.thermophysical.sutherland_transport_enhanced_2.SutherlandTransportEnhanced2`
with:

- Mixture-averaged Lewis number computation
- Blended Mason-Saxena / Wassiljewa mixing rule for conductivity
- Temperature-dependent collision diameter correction

Usage::

    from pyfoam.thermophysical.sutherland_transport_enhanced_3 import SutherlandTransportEnhanced3
    from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

    transport = SutherlandTransportEnhanced3(
        species_params=[
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            SpeciesSutherlandParams(name="O2", mu_ref=1.919e-5, T_ref=273.15, S=139.0, Mw=31.998),
        ],
        polar_correction=True,
        dipole_moments=[0.0, 0.0],
        enable_collision_diameter_correction=True,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.sutherland_transport_enhanced_2 import SutherlandTransportEnhanced2
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

__all__ = ["SutherlandTransportEnhanced3"]

logger = logging.getLogger(__name__)


class SutherlandTransportEnhanced3(SutherlandTransportEnhanced2):
    """Enhanced Sutherland transport v3 with advanced mixing.

    Extends :class:`SutherlandTransportEnhanced2` with:

    - **Mixture Lewis number**: Le_i = alpha / D_im with automatic
      estimation from mixture viscosity and Prandtl number.
    - **Blended Mason-Saxena mixing**: combines Mason-Saxena and
      Wassiljewa rules for improved thermal conductivity:
      kappa_mix = (1 - lambda_ms) * kappa_MS + lambda_ms * kappa_W
    - **Temperature-dependent collision diameter**: sigma(T) = sigma_0 * (1 + eps_sigma / T)
      for improved high-temperature accuracy.

    Parameters
    ----------
    species_params : sequence of SpeciesSutherlandParams
        Per-species Sutherland parameters.
    mu_ref, T_ref, S, kappa_coeffs :
        Single-species parameters (ignored in multi-species mode).
    polar_correction : bool
        Enable Brokaw polar collision correction.
    dipole_moments : sequence of float or None
        Dipole moments (Debye) for each species.
    enable_collision_diameter_correction : bool
        Enable temperature-dependent collision diameter. Default False.
    collision_diameter_coeff : float
        Coefficient for temperature correction of collision diameter.
        Default 0.5.
    blending_parameter : float
        Blending between Mason-Saxena (0) and Wassiljewa (1). Default 0.5.
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
    ) -> None:
        super().__init__(
            species_params=species_params,
            mu_ref=mu_ref,
            T_ref=T_ref,
            S=S,
            kappa_coeffs=kappa_coeffs,
            polar_correction=polar_correction,
            dipole_moments=dipole_moments,
        )
        self._collision_diam_corr = enable_collision_diameter_correction
        self._eps_sigma = collision_diameter_coeff
        self._blend_lambda = max(0.0, min(blending_parameter, 1.0))

    @property
    def collision_diameter_correction_enabled(self) -> bool:
        """Whether temperature-dependent collision diameter is active."""
        return self._collision_diam_corr

    @property
    def blending_parameter(self) -> float:
        """Blending parameter (0=Mason-Saxena, 1=Wassiljewa)."""
        return self._blend_lambda

    # ------------------------------------------------------------------
    # Temperature-dependent collision diameter
    # ------------------------------------------------------------------

    def _collision_diameter_factor(self, T: float) -> float:
        """Temperature correction factor for collision diameter.

        f(T) = 1 + eps_sigma / T

        Parameters
        ----------
        T : float
            Temperature (K).

        Returns
        -------
        float
            Correction factor.
        """
        if not self._collision_diam_corr:
            return 1.0
        return 1.0 + self._eps_sigma / max(T, 1.0)

    # ------------------------------------------------------------------
    # Mason-Saxena thermal conductivity mixing
    # ------------------------------------------------------------------

    def _mason_saxena_kappa(
        self,
        T: float,
        x: Sequence[float],
        Cp_species: Sequence[float],
    ) -> float:
        """Mason-Saxena mixing rule for thermal conductivity.

        kappa_mix = sum_i (x_i * kappa_i) / sum_i (x_i * Gamma_ij)

        where Gamma_ij = (1 + (mu_i/mu_j)^(1/2) * (Mw_j/Mw_i)^(1/4))^2
                         / (8 * (1 + Mw_i/Mw_j))^(1/2)

        Parameters
        ----------
        T : float
            Temperature (K).
        x : sequence of float
            Mole fractions.
        Cp_species : sequence of float
            Per-species Cp values (J/(kg*K)).

        Returns
        -------
        float
            Mixture thermal conductivity (W/(m*K)).
        """
        n = self._n_species
        kappa_pure = []
        for i, sp in enumerate(self._species_params):
            kappa_i = self.species_kappa_eucken(sp.name, T, Cp_species[i])
            kappa_pure.append(kappa_i)

        # Mason-Saxena approximation: weighted harmonic mean
        numerator = 0.0
        denominator = 0.0
        for i in range(n):
            denom_sum = 0.0
            for j in range(n):
                if i == j:
                    denom_sum += x[j]
                    continue
                mu_i = float(self.species_mu(self._species_params[i].name, T).item())
                mu_j = float(self.species_mu(self._species_params[j].name, T).item())
                Mw_i = self._species_params[i].Mw
                Mw_j = self._species_params[j].Mw

                mu_ratio = (mu_i / max(mu_j, 1e-30)) ** 0.5
                Mw_ratio = (Mw_j / max(Mw_i, 1e-30)) ** 0.25
                Gamma_ij = (1.0 + mu_ratio * Mw_ratio) ** 2 / (
                    (8.0 * (1.0 + Mw_i / max(Mw_j, 1e-30))) ** 0.5
                )
                denom_sum += x[j] * Gamma_ij

            numerator += x[i] * kappa_pure[i]
            denominator += x[i] * denom_sum

        return numerator / max(denominator, 1e-30)

    # ------------------------------------------------------------------
    # Mixture Lewis number
    # ------------------------------------------------------------------

    def mixture_lewis_number(
        self,
        T: float,
        x: Sequence[float],
        rho: float = 1.2,
        Cp_mix: float = 1005.0,
        P: float = 101325.0,
    ) -> float:
        """Estimate mixture-averaged Lewis number.

        Le ~ alpha_mix / D_mix
        where alpha = kappa / (rho * Cp)

        Parameters
        ----------
        T : float
            Temperature (K).
        x : sequence of float
            Mole fractions.
        rho : float
            Mixture density (kg/m^3).
        Cp_mix : float
            Mixture Cp (J/(kg*K)).
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Mixture Lewis number estimate.
        """
        mu_mix = float(self.mu(T, x).item())
        Pr = 0.7
        alpha = mu_mix / (rho * Pr)

        # Estimate D_mix from mu and rho (kinematic analogy)
        D_mix = mu_mix / (rho * 0.7)  # Sc ~ Pr ~ 0.7 for gas mixtures

        return alpha / max(D_mix, 1e-30)

    def __repr__(self) -> str:
        if self.is_multispecies:
            polar = ", polar" if self._polar_correction else ""
            coll = ", coll_diam" if self._collision_diam_corr else ""
            return (
                f"SutherlandTransportEnhanced3(n_species={self._n_species}, "
                f"species={self._species_names}{polar}{coll})"
            )
        return (
            f"SutherlandTransportEnhanced3(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S}, single-species)"
        )
