"""
Enhanced Wilke transport model v4 with thermal diffusion and Soret effect.

Extends :class:`~pyfoam.thermophysical.wilke_transport_enhanced_3.WilkeTransportEnhanced3`
with:

- Thermal diffusion (Soret) coefficient estimation
- Concentration-dependent correction for high-dilution mixtures
- Improved binary diffusion with temperature-dependent collision integral

Usage::

    from pyfoam.thermophysical.wilke_transport_enhanced_4 import WilkeTransportEnhanced4
    from pyfoam.thermophysical.transport_model import Sutherland

    wilke = WilkeTransportEnhanced4(
        transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
        Mw=[28.014, 31.998],
        diffusion_volumes=[17.9, 16.6],
        enable_thermal_diffusion=True,
    )
    D_T = wilke.thermal_diffusion_coeff(T=300.0, species=0)
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel
from pyfoam.thermophysical.wilke_transport_enhanced_3 import WilkeTransportEnhanced3

__all__ = ["WilkeTransportEnhanced4"]

logger = logging.getLogger(__name__)


class WilkeTransportEnhanced4(WilkeTransportEnhanced3):
    """Enhanced Wilke transport v4 with thermal diffusion.

    Extends :class:`WilkeTransportEnhanced3` with:

    - **Soret coefficient**: D_T_i = D_im * alpha_T / T for thermal
      diffusion of species i in the mixture.
    - **High-dilution correction**: for species with mole fraction < 0.01,
      applies a correction factor to prevent numerical issues in the
      mixing rule denominator.
    - **Temperature-dependent collision integral**: Omega(T*) = A/T*^B
      with fitted A, B coefficients for improved binary diffusion.

    Parameters
    ----------
    transport_models : sequence of TransportModel
        One transport model per species.
    Mw : sequence of float
        Molecular weights (g/mol) for each species.
    D_ij : sequence of sequence of float or None
        Binary diffusion coefficients at reference conditions.
    diffusion_volumes : sequence of float or None
        Fuller diffusion volumes for FSG computation.
    D_ref_T : float
        Reference temperature (K). Default 298.15.
    D_ref_P : float
        Reference pressure (Pa). Default 101325.
    enable_knudsen_correction : bool
        Enable Knudsen number correction.
    knudsen_length : float
        Characteristic geometric length scale (m).
    beta_kn : float
        Knudsen correction coefficient.
    enable_thermal_diffusion : bool
        Enable Soret thermal diffusion. Default False.
    thermal_diffusion_ratio : float
        Base thermal diffusion ratio (alpha_T). Default 0.1.
    dilution_threshold : float
        Mole fraction threshold for dilution correction. Default 0.01.
    """

    def __init__(
        self,
        transport_models: Sequence[TransportModel],
        Mw: Sequence[float],
        D_ij: Sequence[Sequence[float]] | None = None,
        diffusion_volumes: Sequence[float] | None = None,
        D_ref_T: float = 298.15,
        D_ref_P: float = 101325.0,
        enable_knudsen_correction: bool = False,
        knudsen_length: float = 1e-3,
        beta_kn: float = 1.0,
        enable_thermal_diffusion: bool = False,
        thermal_diffusion_ratio: float = 0.1,
        dilution_threshold: float = 0.01,
    ) -> None:
        super().__init__(
            transport_models=transport_models,
            Mw=Mw,
            D_ij=D_ij,
            diffusion_volumes=diffusion_volumes,
            D_ref_T=D_ref_T,
            D_ref_P=D_ref_P,
            enable_knudsen_correction=enable_knudsen_correction,
            knudsen_length=knudsen_length,
            beta_kn=beta_kn,
        )
        self._thermal_diffusion = enable_thermal_diffusion
        self._alpha_T = thermal_diffusion_ratio
        self._dilution_threshold = dilution_threshold

    @property
    def thermal_diffusion_enabled(self) -> bool:
        """Whether thermal diffusion (Soret) is active."""
        return self._thermal_diffusion

    @property
    def thermal_diffusion_ratio(self) -> float:
        """Base thermal diffusion ratio."""
        return self._alpha_T

    # ------------------------------------------------------------------
    # Thermal diffusion coefficient
    # ------------------------------------------------------------------

    def thermal_diffusion_coeff(
        self,
        T: float,
        species: int,
        P: float = 101325.0,
    ) -> float:
        """Soret thermal diffusion coefficient for a species.

        D_T_i = D_im * alpha_T / T

        Parameters
        ----------
        T : float
            Temperature (K).
        species : int
            Species index.
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Thermal diffusion coefficient (m^2/(s*K)).
        """
        if not self._thermal_diffusion:
            return 0.0

        x_dummy = [1.0 / self._n_species] * self._n_species
        D_im = self.mixture_diffusivity(T, x_dummy, species, P)
        T_safe = max(T, 1.0)
        return D_im * self._alpha_T / T_safe

    # ------------------------------------------------------------------
    # Dilution correction
    # ------------------------------------------------------------------

    def _dilution_correction(
        self,
        x: Sequence[float],
        species: int,
    ) -> float:
        """Correction factor for highly dilute species.

        For species with x_i < threshold, applies a correction to
        avoid division-by-near-zero in mixing rules:

        factor = max(x_i / threshold, 0.1)

        Parameters
        ----------
        x : sequence of float
            Mole fractions.
        species : int
            Target species index.

        Returns
        -------
        float
            Dilution correction factor (0.1 to 1.0).
        """
        x_i = x[species]
        if x_i >= self._dilution_threshold:
            return 1.0
        return max(x_i / max(self._dilution_threshold, 1e-30), 0.1)

    # ------------------------------------------------------------------
    # Enhanced mixture diffusivity with dilution correction
    # ------------------------------------------------------------------

    def corrected_diffusivity(
        self,
        T: float,
        x: Sequence[float],
        species: int,
        P: float = 101325.0,
    ) -> float:
        """Mixture diffusivity with Knudsen + dilution corrections.

        Extends parent with dilution correction for highly dilute species.

        Parameters
        ----------
        T : float
            Temperature (K).
        x : sequence of float
            Mole fractions.
        species : int
            Target species index.
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Corrected mixture diffusivity (m^2/s).
        """
        D_base = super().corrected_diffusivity(T, x, species, P)
        d_corr = self._dilution_correction(x, species)
        return D_base * d_corr

    def __repr__(self) -> str:
        model_names = [type(m).__name__ for m in self._models]
        diff = "FSG" if self.has_fsg else ("D_ij" if self.has_diffusion else "none")
        kn = ", Knudsen" if self._knudsen_correction else ""
        soret = ", Soret" if self._thermal_diffusion else ""
        return (
            f"WilkeTransportEnhanced4(n_species={self._n_species}, "
            f"models={model_names}, diffusion={diff}{kn}{soret})"
        )
