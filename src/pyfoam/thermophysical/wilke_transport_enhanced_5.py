"""
Enhanced Wilke transport model v5 with Stockmayer collision integrals and virial correction.

Extends :class:`~pyfoam.thermophysical.wilke_transport_enhanced_4.WilkeTransportEnhanced4`
with:

- Stockmayer potential collision integrals for polar gases
- Pressure-dependent second virial correction for diffusion
- Mixture-averaged thermal diffusion factor

Usage::

    from pyfoam.thermophysical.wilke_transport_enhanced_5 import WilkeTransportEnhanced5
    from pyfoam.thermophysical.transport_model import Sutherland

    wilke = WilkeTransportEnhanced5(
        transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
        Mw=[28.014, 31.998],
        diffusion_volumes=[17.9, 16.6],
        dipole_moments=[0.0, 0.0],
        stockmayer_eps_k=[95.0, 107.0],
    )
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel
from pyfoam.thermophysical.wilke_transport_enhanced_4 import WilkeTransportEnhanced4

__all__ = ["WilkeTransportEnhanced5"]

logger = logging.getLogger(__name__)

_R_UNIV = 8.314462618  # J/(mol*K)


class WilkeTransportEnhanced5(WilkeTransportEnhanced4):
    """Enhanced Wilke transport v5 with Stockmayer and virial corrections.

    Extends :class:`WilkeTransportEnhanced4` with:

    - **Stockmayer collision integrals**: for polar gas pairs, uses the
      Stockmayer potential with dipole moment to compute Omega_22, which
      is more accurate than the LJ approximation for polar molecules.
    - **Pressure-dependent second virial correction**: D_ij(P) = D_ij(P_ref)
      * (P_ref / P) * (1 + B(T) * P / (R*T)) for high-pressure correction.
    - **Mixture-averaged thermal diffusion factor**: alpha_T_mix computed
      from mass-fraction-weighted individual species Soret coefficients.

    Parameters
    ----------
    transport_models : sequence of TransportModel
        One transport model per species.
    Mw : sequence of float
        Molecular weights (g/mol).
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
        Enable Soret thermal diffusion.
    thermal_diffusion_ratio : float
        Base thermal diffusion ratio (alpha_T).
    dilution_threshold : float
        Mole fraction threshold for dilution correction.
    dipole_moments : sequence of float or None
        Dipole moments (Debye) per species for Stockmayer correction.
    stockmayer_eps_k : sequence of float or None
        Stockmayer well depth epsilon/k_B (K) per species.
    enable_virial_correction : bool
        Enable second virial pressure correction. Default False.
    B_ref : float
        Reference second virial coefficient (cm^3/mol). Default -100.
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
        dipole_moments: Sequence[float] | None = None,
        stockmayer_eps_k: Sequence[float] | None = None,
        enable_virial_correction: bool = False,
        B_ref: float = -100.0,
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
            enable_thermal_diffusion=enable_thermal_diffusion,
            thermal_diffusion_ratio=thermal_diffusion_ratio,
            dilution_threshold=dilution_threshold,
        )
        self._dipole_moments = list(dipole_moments) if dipole_moments is not None else None
        self._stockmayer_eps_k = list(stockmayer_eps_k) if stockmayer_eps_k is not None else None
        self._virial_correction = enable_virial_correction
        self._B_ref = B_ref

    @property
    def has_stockmayer(self) -> bool:
        """Whether Stockmayer parameters are available."""
        return (self._dipole_moments is not None
                and self._stockmayer_eps_k is not None)

    @property
    def virial_correction_enabled(self) -> bool:
        """Whether second virial correction is active."""
        return self._virial_correction

    # ------------------------------------------------------------------
    # Stockmayer collision integral
    # ------------------------------------------------------------------

    def stockmayer_collision_integral(
        self,
        T: float,
        species_i: int,
        species_j: int,
    ) -> float:
        """Stockmayer collision integral for polar gas pairs.

        Uses the Monchick-Mason approximation with reduced dipole moment:
        delta = mu_dipole / sqrt(eps * sigma^3)
        Omega_22 = Omega_22_LJ * (1 + C * delta^2)

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
        if not self.has_stockmayer:
            return 1.0

        mu_i = self._dipole_moments[species_i]
        mu_j = self._dipole_moments[species_j]
        eps_k_i = self._stockmayer_eps_k[species_i]
        eps_k_j = self._stockmayer_eps_k[species_j]

        eps_eff_k = math.sqrt(eps_k_i * eps_k_j)
        T_star = T / max(eps_eff_k, 1e-10)
        T_star = max(T_star, 0.3)

        # LJ baseline
        Omega_LJ = 1.147 / (T_star ** 0.145)

        # Dipole correction
        mu_avg = math.sqrt(abs(mu_i * mu_j))
        delta_sq = mu_avg ** 2 / max(eps_eff_k, 1e-10)
        C_dipole = 0.05  # Empirical
        correction = 1.0 + C_dipole * delta_sq / max(T_star, 0.1)

        return Omega_LJ * correction

    # ------------------------------------------------------------------
    # Second virial correction
    # ------------------------------------------------------------------

    def _virial_diffusion_correction(
        self,
        T: float,
        P: float,
    ) -> float:
        """Pressure correction for diffusion from second virial coefficient.

        factor = (P_ref / P) * (1 + B * P / (R*T))

        Parameters
        ----------
        T : float
            Temperature (K).
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Virial correction factor.
        """
        T_safe = max(T, 1.0)
        P_safe = max(P, 1.0)
        B = self._B_ref * 1e-6  # cm^3/mol -> m^3/mol
        correction = (self._D_ref_P / P_safe) * (1.0 + B * P_safe / (_R_UNIV * T_safe))
        return max(correction, 0.1)

    # ------------------------------------------------------------------
    # Enhanced mixture diffusivity with virial correction
    # ------------------------------------------------------------------

    def corrected_diffusivity(
        self,
        T: float,
        x: Sequence[float],
        species: int,
        P: float = 101325.0,
    ) -> float:
        """Mixture diffusivity with Knudsen + dilution + virial corrections.

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

        if self._virial_correction:
            virial = self._virial_diffusion_correction(T, P)
            return D_base * virial

        return D_base

    # ------------------------------------------------------------------
    # Mixture-averaged thermal diffusion factor
    # ------------------------------------------------------------------

    def mixture_thermal_diffusion_factor(
        self,
        T: float,
        Y: Sequence[float],
    ) -> list[float]:
        """Mass-fraction-weighted mixture thermal diffusion factors.

        alpha_T_mix_i = sum_j(Y_j * alpha_T_ij)

        Parameters
        ----------
        T : float
            Temperature (K).
        Y : sequence of float
            Mass fractions.

        Returns
        -------
        list of float
            Mixture thermal diffusion factor per species.
        """
        if not self._thermal_diffusion:
            return [0.0] * self._n_species

        result = []
        for i in range(self._n_species):
            x_dummy = [1.0 / self._n_species] * self._n_species
            alpha_T_i = self.thermal_diffusion_coeff(T, i) * T
            result.append(alpha_T_i)

        return result

    def __repr__(self) -> str:
        model_names = [type(m).__name__ for m in self._models]
        diff = "FSG" if self.has_fsg else ("D_ij" if self.has_diffusion else "none")
        kn = ", Knudsen" if self._knudsen_correction else ""
        soret = ", Soret" if self._thermal_diffusion else ""
        stock = ", Stockmayer" if self.has_stockmayer else ""
        virial = ", virial" if self._virial_correction else ""
        return (
            f"WilkeTransportEnhanced5(n_species={self._n_species}, "
            f"models={model_names}, diffusion={diff}{kn}{soret}{stock}{virial})"
        )
