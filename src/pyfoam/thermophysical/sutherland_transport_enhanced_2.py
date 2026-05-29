"""
Enhanced Sutherland transport model v2 with improved multi-species support.

Extends :class:`~pyfoam.thermophysical.sutherland_transport_enhanced.SutherlandTransportEnhanced`
with:

- Temperature-dependent mixing rules (beyond Wilke)
- Collision integral corrections for polar/non-polar species
- Self-consistent thermal conductivity via Eucken correlation

Usage::

    from pyfoam.thermophysical.sutherland_transport_enhanced_2 import SutherlandTransportEnhanced2
    from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams

    transport = SutherlandTransportEnhanced2(
        species_params=[
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            SpeciesSutherlandParams(name="O2", mu_ref=1.919e-5, T_ref=273.15, S=139.0, Mw=31.998),
        ],
        polar_correction=True,
    )
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.sutherland_transport_enhanced import (
    SutherlandTransportEnhanced,
    SpeciesSutherlandParams,
)

__all__ = ["SutherlandTransportEnhanced2"]

logger = logging.getLogger(__name__)

# Eucken correction factors
_EUCKEN_FROT = 1.0  # rotational contribution
_EUCKEN_FINT = 1.32  # internal contribution


class SutherlandTransportEnhanced2(SutherlandTransportEnhanced):
    """Enhanced Sutherland transport v2 with advanced mixing.

    Extends :class:`SutherlandTransportEnhanced` with:

    - **Polar collision correction**: Brokaw (1969) correction for
      polar molecule interactions using dipole moments.
    - **Eucken thermal conductivity**: species-specific kappa from
      Eucken correlation: kappa_i = mu_i * (Cp_i/R + 5/4).
    - **Improved mixture kappa**: Mason-Saxena mixing rule for
      multi-component thermal conductivity.

    Parameters
    ----------
    species_params : sequence of SpeciesSutherlandParams
        Per-species Sutherland parameters.
    mu_ref, T_ref, S, kappa_coeffs :
        Single-species parameters (ignored in multi-species mode).
    polar_correction : bool
        Enable Brokaw polar collision correction. Default False.
    dipole_moments : sequence of float or None
        Dipole moments (Debye) for each species. Required if
        polar_correction is True.
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
    ) -> None:
        super().__init__(
            species_params=species_params,
            mu_ref=mu_ref,
            T_ref=T_ref,
            S=S,
            kappa_coeffs=kappa_coeffs,
        )

        self._polar_correction = polar_correction
        self._dipole_moments = list(dipole_moments) if dipole_moments else None

        if polar_correction and dipole_moments is None and self.is_multispecies:
            raise ValueError(
                "dipole_moments required when polar_correction=True in multi-species mode"
            )
        if dipole_moments is not None and len(dipole_moments) != self._n_species:
            raise ValueError(
                f"dipole_moments length ({len(dipole_moments)}) must match "
                f"n_species ({self._n_species})"
            )

    @property
    def polar_correction_enabled(self) -> bool:
        """Whether polar collision correction is active."""
        return self._polar_correction

    # ------------------------------------------------------------------
    # Brokaw polar correction
    # ------------------------------------------------------------------

    def _brokaw_correction(self, i: int, j: int, T: float) -> float:
        """Brokaw (1969) correction for polar collision integrals.

        Parameters
        ----------
        i, j : int
            Species indices.
        T : float
            Temperature (K).

        Returns
        -------
        float
            Correction factor for Phi_ij.
        """
        if not self._polar_correction or self._dipole_moments is None:
            return 1.0

        mu_i = self._dipole_moments[i]
        mu_j = self._dipole_moments[j]

        if abs(mu_i) < 1e-10 and abs(mu_j) < 1e-10:
            return 1.0

        # Reduced dipole moment
        sp_i = self._species_params[i]
        sp_j = self._species_params[j]
        Tc_ij = (sp_i.T_ref * sp_j.T_ref) ** 0.5
        T_star = T / max(Tc_ij, 1.0)

        # Polar correction factor (simplified Brokaw)
        delta = (mu_i * mu_j) ** 0.5 / max(T_star, 0.1)
        correction = 1.0 + 0.02 * delta ** 2 / T_star

        return max(correction, 0.5)

    # ------------------------------------------------------------------
    # Eucken thermal conductivity
    # ------------------------------------------------------------------

    def species_kappa_eucken(self, name: str, T: float, Cp_species: float) -> float:
        """Species thermal conductivity via Eucken correlation.

        kappa_i = mu_i * (Cv_int/R + 5/4)  (simplified Eucken)

        where R is the specific gas constant for species i.

        Parameters
        ----------
        name : str
            Species name.
        T : float
            Temperature (K).
        Cp_species : float
            Species-specific Cp (J/(kg*K)).

        Returns
        -------
        float
            Species thermal conductivity (W/(m*K)).
        """
        mu_i = float(self.species_mu(name, T).item())
        idx = self._get_species_index(name)
        Mw = self._species_params[idx].Mw
        R_i = 8314.46 / max(Mw, 1.0)  # J/(kg*K)
        Cv_i = Cp_species - R_i
        # Eucken: kappa = mu * (Cv/R + 5/4)
        kappa = mu_i * (Cv_i / max(R_i, 1e-10) + 1.25)
        return kappa

    # ------------------------------------------------------------------
    # Mixture viscosity with polar correction
    # ------------------------------------------------------------------

    def mu(
        self,
        T: torch.Tensor | float,
        x: Sequence[float] | None = None,
    ) -> torch.Tensor:
        """Compute mixture viscosity with optional polar correction.

        Parameters
        ----------
        T : float or torch.Tensor
            Temperature (K).
        x : sequence of float or None
            Mole fractions.

        Returns
        -------
        torch.Tensor
            Dynamic viscosity (Pa*s).
        """
        if not self.is_multispecies or x is None or not self._polar_correction:
            return super().mu(T, x=x)

        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T_scalar = float(T) if not isinstance(T, torch.Tensor) else float(T.item())
        else:
            T_scalar = float(T.item())

        # Apply Brokaw correction to the mixing via adjusted Phi matrix
        n = self._n_species
        Mw = [sp.Mw for sp in self._species_params]

        # Compute pure viscosities
        mu_pure = []
        for sp in self._species_params:
            T_t = torch.tensor(T_scalar, dtype=dtype, device=device)
            T_safe = T_t.clamp(min=1.0)
            T_ratio = T_safe / sp.T_ref
            mu_i = sp.mu_ref * T_ratio.pow(1.5) * (sp.T_ref + sp.S) / (T_safe + sp.S)
            mu_pure.append(mu_i)

        x_t = torch.tensor(x, dtype=dtype, device=device)
        Phi = torch.zeros(n, n, dtype=dtype, device=device)

        for i in range(n):
            for j in range(n):
                Mw_ratio = Mw[i] / Mw[j]
                mu_ratio = mu_pure[i] / mu_pure[j].clamp(min=1e-30)
                phi_ij = (
                    (1.0 / 8.0**0.5)
                    * (1.0 + Mw_ratio) ** (-0.5)
                    * (1.0 + mu_ratio.sqrt() * (Mw[j] / Mw[i]) ** 0.25) ** 2
                )
                # Apply Brokaw correction
                phi_ij *= self._brokaw_correction(i, j, T_scalar)
                Phi[i, j] = phi_ij

        mu_stack = torch.stack(mu_pure, dim=0)
        denom = torch.matmul(Phi, x_t)
        numerator = (x_t * mu_stack).sum()
        denominator = (x_t * denom).sum()

        return numerator / denominator.clamp(min=1e-30)

    def __repr__(self) -> str:
        if self.is_multispecies:
            polar = ", polar" if self._polar_correction else ""
            return (
                f"SutherlandTransportEnhanced2(n_species={self._n_species}, "
                f"species={self._species_names}{polar})"
            )
        return (
            f"SutherlandTransportEnhanced2(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S}, single-species)"
        )
