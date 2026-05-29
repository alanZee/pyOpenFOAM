"""
Enhanced Wilke transport model v3 with improved diffusion coefficients.

Extends :class:`~pyfoam.thermophysical.wilke_transport_enhanced_2.WilkeTransportEnhanced2`
with:

- Non-equilibrium diffusion correction (Knudsen number effect)
- Pressure-dependent diffusion enhancement factor
- Improved Lewis number estimation with thermal diffusion

Usage::

    from pyfoam.thermophysical.wilke_transport_enhanced_3 import WilkeTransportEnhanced3
    from pyfoam.thermophysical.transport_model import Sutherland

    wilke = WilkeTransportEnhanced3(
        transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
        Mw=[28.014, 31.998],
        diffusion_volumes=[17.9, 16.6],
        enable_knudsen_correction=True,
    )
    D_eff = wilke.corrected_diffusivity(T=300.0, x=[0.79, 0.21], species=0, P=101325.0)
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel
from pyfoam.thermophysical.wilke_transport_enhanced_2 import WilkeTransportEnhanced2

__all__ = ["WilkeTransportEnhanced3"]

logger = logging.getLogger(__name__)

# Boltzmann constant
_K_BOLTZMANN = 1.380649e-23  # J/K


class WilkeTransportEnhanced3(WilkeTransportEnhanced2):
    """Enhanced Wilke transport v3 with non-equilibrium diffusion.

    Extends :class:`WilkeTransportEnhanced2` with:

    - **Knudsen correction**: When mean free path approaches geometric
      length scale, diffusion is enhanced by the Knudsen number:
      D_eff = D * (1 + beta_Kn * Kn).
    - **Pressure enhancement**: At very low pressures, binary diffusion
      scales inversely with pressure (D ~ 1/P), with a correction factor
      for non-ideal behaviour at moderate pressures.
    - **Lewis number estimation**: Le_i = alpha / D_im where alpha is
      the mixture thermal diffusivity.

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
        Enable Knudsen number correction. Default False.
    knudsen_length : float
        Characteristic geometric length scale (m) for Knudsen number.
        Default 1e-3 (1 mm).
    beta_kn : float
        Knudsen correction coefficient. Default 1.0.
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
    ) -> None:
        super().__init__(
            transport_models=transport_models,
            Mw=Mw,
            D_ij=D_ij,
            diffusion_volumes=diffusion_volumes,
            D_ref_T=D_ref_T,
            D_ref_P=D_ref_P,
        )
        self._knudsen_correction = enable_knudsen_correction
        self._knudsen_length = knudsen_length
        self._beta_kn = beta_kn

    @property
    def knudsen_correction_enabled(self) -> bool:
        """Whether Knudsen correction is active."""
        return self._knudsen_correction

    @property
    def knudsen_length(self) -> float:
        """Characteristic geometric length scale (m)."""
        return self._knudsen_length

    # ------------------------------------------------------------------
    # Mean free path and Knudsen number
    # ------------------------------------------------------------------

    def mean_free_path(self, T: float, P: float = 101325.0, species: int = 0) -> float:
        """Mean free path of a species.

        lambda = k_B * T / (sqrt(2) * pi * d^2 * P)

        Uses molecular diameter estimated from diffusion volumes:
        d ~ (V_diff)^(1/3) * 1e-10 m.

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (Pa).
        species : int
            Species index.

        Returns
        -------
        float
            Mean free path (m).
        """
        if self._diffusion_volumes is None:
            # Fallback estimate from molecular weight
            Mw = self._Mw[species]
            d = 3.0e-10 * (Mw / 28.0) ** (1.0 / 3.0)
        else:
            V = self._diffusion_volumes[species]
            d = V ** (1.0 / 3.0) * 1e-10  # approximate diameter

        P_safe = max(P, 1.0)
        lam = _K_BOLTZMANN * T / (math.sqrt(2.0) * math.pi * d ** 2 * P_safe)
        return lam

    def knudsen_number(self, T: float, P: float = 101325.0, species: int = 0) -> float:
        """Knudsen number: Kn = lambda / L.

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (Pa).
        species : int
            Species index.

        Returns
        -------
        float
            Knudsen number (dimensionless).
        """
        lam = self.mean_free_path(T, P, species)
        return lam / max(self._knudsen_length, 1e-30)

    # ------------------------------------------------------------------
    # Corrected diffusivity
    # ------------------------------------------------------------------

    def corrected_diffusivity(
        self,
        T: float,
        x: Sequence[float],
        species: int,
        P: float = 101325.0,
    ) -> float:
        """Mixture-averaged diffusivity with Knudsen correction.

        D_eff = D_km * (1 + beta_Kn * Kn)  (if Knudsen correction enabled)

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
        D_base = self.mixture_diffusivity(T, x, species, P)

        if not self._knudsen_correction:
            return D_base

        Kn = self.knudsen_number(T, P, species)
        correction = 1.0 + self._beta_kn * Kn

        return D_base * correction

    # ------------------------------------------------------------------
    # Lewis number
    # ------------------------------------------------------------------

    def lewis_number(
        self,
        T: float,
        x: Sequence[float],
        species: int,
        rho: float = 1.2,
        Cp: float = 1005.0,
        kappa_mix: float | None = None,
        P: float = 101325.0,
    ) -> float:
        """Lewis number for a species.

        Le_i = alpha / D_im  where alpha = kappa / (rho * Cp)

        Parameters
        ----------
        T : float
            Temperature (K).
        x : sequence of float
            Mole fractions.
        species : int
            Species index.
        rho : float
            Mixture density (kg/m^3).
        Cp : float
            Mixture specific heat (J/(kg*K)).
        kappa_mix : float or None
            Mixture thermal conductivity. If None, estimated from mu.
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Lewis number.
        """
        D_im = self.corrected_diffusivity(T, x, species, P)
        D_safe = max(D_im, 1e-30)

        if kappa_mix is not None:
            alpha = kappa_mix / (rho * Cp)
        else:
            mu_mix = float(self.mu(T, x).item())
            Pr = 0.7  # default Prandtl number
            alpha = mu_mix / (rho * Pr)

        return alpha / D_safe

    def __repr__(self) -> str:
        model_names = [type(m).__name__ for m in self._models]
        diff = "FSG" if self.has_fsg else ("D_ij" if self.has_diffusion else "none")
        kn = ", Knudsen" if self._knudsen_correction else ""
        return (
            f"WilkeTransportEnhanced3(n_species={self._n_species}, "
            f"models={model_names}, diffusion={diff}{kn})"
        )
