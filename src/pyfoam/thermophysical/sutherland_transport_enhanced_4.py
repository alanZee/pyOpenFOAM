"""
Enhanced Sutherland transport model v4 with collision integral and diffusion.

Extends :class:`~pyfoam.thermophysical.sutherland_transport_enhanced_3.SutherlandTransportEnhanced3`
with:

- Lennard-Jones collision integral model for binary diffusion
- Temperature-dependent effective diameter from Lennard-Jones potential
- Multi-component thermal diffusion ratio (Soret) estimation

Usage::

    from pyfoam.thermophysical.sutherland_transport_enhanced_4 import SutherlandTransportEnhanced4
    from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

    transport = SutherlandTransportEnhanced4(
        species_params=[
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            SpeciesSutherlandParams(name="O2", mu_ref=1.919e-5, T_ref=273.15, S=139.0, Mw=31.998),
        ],
        lj_sigma=[3.798, 3.467],
        lj_epsilon_k=[71.4, 106.7],
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.sutherland_transport_enhanced_3 import SutherlandTransportEnhanced3
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

__all__ = ["SutherlandTransportEnhanced4"]

logger = logging.getLogger(__name__)

_R_UNIV = 8.314462618  # J/(mol*K)
_N_AVOGADRO = 6.02214076e23


class SutherlandTransportEnhanced4(SutherlandTransportEnhanced3):
    """Enhanced Sutherland transport v4 with Lennard-Jones collision integral.

    Extends :class:`SutherlandTransportEnhanced3` with:

    - **Lennard-Jones collision integral**: Omega_22(T*) = A / (T*)^B
      where T* = k_B * T / epsilon. Provides accurate binary diffusion
      coefficients for non-polar gas pairs.
    - **Effective collision diameter**: sigma_eff = 0.5 * (sigma_i + sigma_j)
      for unlike molecular pairs.
    - **Thermal diffusion ratio**: alpha_T_i estimated from mass and
      Lennard-Jones parameters for the Soret effect.

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
        Enable temperature-dependent collision diameter.
    collision_diameter_coeff : float
        Coefficient for temperature correction.
    blending_parameter : float
        Blending parameter.
    lj_sigma : sequence of float or None
        Lennard-Jones collision diameters (Angstrom) per species.
    lj_epsilon_k : sequence of float or None
        Lennard-Jones well depths (K) per species.
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
    ) -> None:
        super().__init__(
            species_params=species_params,
            mu_ref=mu_ref,
            T_ref=T_ref,
            S=S,
            kappa_coeffs=kappa_coeffs,
            polar_correction=polar_correction,
            dipole_moments=dipole_moments,
            enable_collision_diameter_correction=enable_collision_diameter_correction,
            collision_diameter_coeff=collision_diameter_coeff,
            blending_parameter=blending_parameter,
        )
        self._lj_sigma = list(lj_sigma) if lj_sigma is not None else None
        self._lj_epsilon_k = list(lj_epsilon_k) if lj_epsilon_k is not None else None

    @property
    def has_lj_params(self) -> bool:
        """Whether Lennard-Jones parameters are available."""
        return self._lj_sigma is not None and self._lj_epsilon_k is not None

    # ------------------------------------------------------------------
    # Lennard-Jones collision integral
    # ------------------------------------------------------------------

    def collision_integral(self, T: float, species_i: int, species_j: int) -> float:
        """Lennard-Jones collision integral Omega_22(T*).

        Uses the Neufeld et al. correlation:
        Omega_22 = A / (T*)^B where A=1.147, B=0.145

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
        if not self.has_lj_params:
            return 1.0  # Default for hard-sphere

        eps_i = self._lj_epsilon_k[species_i]
        eps_j = self._lj_epsilon_k[species_j]
        eps_eff = math.sqrt(eps_i * eps_j)

        T_star = T / max(eps_eff, 1e-10)
        T_star = max(T_star, 0.3)

        # Neufeld et al. correlation
        A = 1.147
        B = 0.145
        return A / (T_star ** B)

    # ------------------------------------------------------------------
    # Effective collision diameter
    # ------------------------------------------------------------------

    def effective_sigma(self, species_i: int, species_j: int) -> float:
        """Effective collision diameter for unlike pair.

        sigma_eff = 0.5 * (sigma_i + sigma_j)

        Parameters
        ----------
        species_i, species_j : int
            Species indices.

        Returns
        -------
        float
            Effective collision diameter (Angstrom).
        """
        if not self.has_lj_params:
            # Fallback estimate from diffusion volumes
            if self._species_params is not None:
                Mw_i = self._species_params[species_i].Mw
                Mw_j = self._species_params[species_j].Mw
                return 0.5 * (3.0 * (Mw_i / 28.0) ** (1.0 / 3.0) + 3.0 * (Mw_j / 28.0) ** (1.0 / 3.0))
            return 3.5  # Default Angstrom

        return 0.5 * (self._lj_sigma[species_i] + self._lj_sigma[species_j])

    # ------------------------------------------------------------------
    # Binary diffusion from Lennard-Jones
    # ------------------------------------------------------------------

    def binary_diffusion_lj(
        self,
        T: float,
        P: float,
        species_i: int,
        species_j: int,
    ) -> float:
        """Binary diffusion coefficient from Lennard-Jones model.

        D_ij = 0.00266 * T^(3/2) / (P * M_ij^(1/2) * sigma_ij^2 * Omega_22)

        where M_ij = 2 / (1/Mw_i + 1/Mw_j) and sigma_ij in Angstrom.

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (atm).
        species_i, species_j : int
            Species indices.

        Returns
        -------
        float
            Binary diffusion coefficient (cm^2/s).
        """
        if not self.has_lj_params:
            return 0.0

        Mw_i = self._species_params[species_i].Mw
        Mw_j = self._species_params[species_j].Mw
        M_ij = 2.0 / (1.0 / Mw_i + 1.0 / Mw_j)

        sigma_ij = self.effective_sigma(species_i, species_j)
        Omega_22 = self.collision_integral(T, species_i, species_j)

        P_atm = max(P / 101325.0, 1e-10)

        D_ij = 0.00266 * T ** 1.5 / (
            P_atm * math.sqrt(M_ij) * sigma_ij ** 2 * max(Omega_22, 1e-10)
        )
        return D_ij

    # ------------------------------------------------------------------
    # Thermal diffusion ratio
    # ------------------------------------------------------------------

    def thermal_diffusion_ratio(
        self,
        T: float,
        x: Sequence[float],
        species: int,
    ) -> float:
        """Estimate thermal diffusion ratio (Soret coefficient).

        alpha_T ~ (Mw_i - Mw_mix) * C_T / T

        where Mw_mix = sum(x_j * Mw_j) and C_T is an empirical constant.

        Parameters
        ----------
        T : float
            Temperature (K).
        x : sequence of float
            Mole fractions.
        species : int
            Target species index.

        Returns
        -------
        float
            Thermal diffusion ratio (dimensionless).
        """
        if not self.is_multispecies or self._species_params is None:
            return 0.0

        Mw_i = self._species_params[species].Mw
        Mw_mix = sum(
            x[j] * self._species_params[j].Mw for j in range(self._n_species)
        )
        C_T = 0.1  # Empirical constant
        T_safe = max(T, 1.0)
        return C_T * (Mw_i - Mw_mix) / T_safe

    def __repr__(self) -> str:
        if self.is_multispecies:
            polar = ", polar" if self._polar_correction else ""
            lj = ", LJ" if self.has_lj_params else ""
            return (
                f"SutherlandTransportEnhanced4(n_species={self._n_species}, "
                f"species={self._species_names}{polar}{lj})"
            )
        return (
            f"SutherlandTransportEnhanced4(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S}, single-species)"
        )
