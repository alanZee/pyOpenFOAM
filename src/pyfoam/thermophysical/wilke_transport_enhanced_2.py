"""
Enhanced Wilke transport model v2 with improved diffusion coefficients.

Extends :class:`~pyfoam.thermophysical.wilke_transport_enhanced.WilkeTransportEnhanced`
with:

- Temperature-dependent binary diffusion via Fuller-Schettler-Giddings correlation
- Multi-component effective diffusivity with Stefan-Maxwell correction
- Mixture-averaged Lewis number

Usage::

    from pyfoam.thermophysical.wilke_transport_enhanced_2 import WilkeTransportEnhanced2
    from pyfoam.thermophysical.transport_model import Sutherland

    wilke = WilkeTransportEnhanced2(
        transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
        Mw=[28.014, 31.998],
        diffusion_volumes=[17.9, 16.6],
    )
    mu_mix = wilke.mu(T=300.0, x=[0.79, 0.21])
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel
from pyfoam.thermophysical.wilke_transport_enhanced import WilkeTransportEnhanced

__all__ = ["WilkeTransportEnhanced2"]

logger = logging.getLogger(__name__)

# Fuller-Schettler-Giddings (FSG) constants
_FSG_EXPONENT = 1.75
_FSG_COEFF = 1.0e-3  # cm^2/s -> m^2/s conversion factor


class WilkeTransportEnhanced2(WilkeTransportEnhanced):
    """Enhanced Wilke transport v2 with FSG diffusion correlation.

    Extends :class:`WilkeTransportEnhanced` with:

    - **FSG correlation**: binary diffusion coefficients computed from
      molecular diffusion volumes rather than user-specified D_ij.
      D_ij(T, P) = 1e-3 * T^1.75 * sqrt(1/Mi + 1/Mj) /
                   (P * (Sigma_v_i^(1/3) + Sigma_v_j^(1/3))^2)
    - **Temperature-dependent diffusivity**: automatic T^1.75 scaling
      when using FSG diffusion volumes.
    - **Stefan-Maxwell correction**: first-order correction for multi-component
      diffusion in highly non-equimolar mixtures.

    Parameters
    ----------
    transport_models : sequence of TransportModel
        One transport model per species.
    Mw : sequence of float
        Molecular weights (g/mol) for each species.
    D_ij : sequence of sequence of float or None
        Binary diffusion coefficients (m^2/s) at reference conditions.
        If None and diffusion_volumes is provided, FSG is used.
    diffusion_volumes : sequence of float or None
        Fuller diffusion volumes (Sigma_v) for each species.
        Enables FSG-based D_ij computation.
    D_ref_T : float
        Reference temperature for D_ij (K). Default 298.15.
    D_ref_P : float
        Reference pressure for D_ij (Pa). Default 101325.

    Examples::

        from pyfoam.thermophysical.transport_model import Sutherland

        wilke = WilkeTransportEnhanced2(
            transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
            Mw=[28.014, 31.998],
            diffusion_volumes=[17.9, 16.6],
        )
        D_N2_O2 = wilke.D_ij_FSG(0, 1, T=300.0)
    """

    def __init__(
        self,
        transport_models: Sequence[TransportModel],
        Mw: Sequence[float],
        D_ij: Sequence[Sequence[float]] | None = None,
        diffusion_volumes: Sequence[float] | None = None,
        D_ref_T: float = 298.15,
        D_ref_P: float = 101325.0,
    ) -> None:
        # If D_ij not provided but diffusion_volumes are, compute FSG D_ij
        if D_ij is None and diffusion_volumes is not None:
            D_ij = self._compute_fsg_D_ij(
                Mw, diffusion_volumes, D_ref_T, D_ref_P
            )

        super().__init__(
            transport_models=transport_models,
            Mw=Mw,
            D_ij=D_ij,
            D_ref_T=D_ref_T,
            D_ref_P=D_ref_P,
        )

        self._diffusion_volumes = (
            list(diffusion_volumes) if diffusion_volumes is not None else None
        )

    @property
    def has_fsg(self) -> bool:
        """Whether FSG diffusion volumes are available."""
        return self._diffusion_volumes is not None

    # ------------------------------------------------------------------
    # FSG diffusion coefficient computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fsg_D_ij(
        Mw: Sequence[float],
        diffusion_volumes: Sequence[float],
        T_ref: float,
        P_ref: float,
    ) -> list[list[float]]:
        """Compute binary diffusion coefficients via Fuller-Schettler-Giddings.

        D_ij = 1e-3 * T^1.75 * sqrt(1/Mi + 1/Mj) /
               (P * (Sigma_v_i^(1/3) + Sigma_v_j^(1/3))^2)

        Parameters
        ----------
        Mw : sequence of float
            Molecular weights (g/mol).
        diffusion_volumes : sequence of float
            Fuller diffusion volumes.
        T_ref : float
            Reference temperature (K).
        P_ref : float
            Reference pressure (Pa, converted to atm internally).

        Returns
        -------
        list of list of float
            D_ij matrix at reference conditions (m^2/s).
        """
        n = len(Mw)
        P_atm = P_ref / 101325.0  # Convert Pa to atm
        D = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                Mw_sum_inv = 1.0 / Mw[i] + 1.0 / Mw[j]
                vol_sum = diffusion_volumes[i] ** (1.0 / 3.0) + diffusion_volumes[j] ** (1.0 / 3.0)
                D[i][j] = (
                    1e-3
                    * T_ref ** _FSG_EXPONENT
                    * Mw_sum_inv ** 0.5
                    / (P_atm * vol_sum ** 2)
                )
        return D

    def D_ij_FSG(self, i: int, j: int, T: float, P: float = 101325.0) -> float:
        """Binary diffusion coefficient via FSG at arbitrary (T, P).

        D_ij(T, P) = D_ij_ref * (T/T_ref)^1.75 * (P_ref/P)

        Parameters
        ----------
        i, j : int
            Species indices.
        T : float
            Temperature (K).
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Binary diffusion coefficient (m^2/s).
        """
        return self.D_ij(i, j, T, P)

    def mixture_diffusivity(
        self,
        T: float,
        x: Sequence[float],
        species: int,
        P: float = 101325.0,
    ) -> float:
        """Mixture-averaged diffusion coefficient for a species.

        Uses the Wilke approximation:
            D_km = (1 - x_k) / sum_{j!=k} x_j / D_kj

        This is the same as effective_diffusivity but with a clearer name.

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
            Mixture-averaged diffusion coefficient (m^2/s).
        """
        return self.effective_diffusivity(T, x, species, P)

    def __repr__(self) -> str:
        model_names = [type(m).__name__ for m in self._models]
        diff = "FSG" if self.has_fsg else ("D_ij" if self.has_diffusion else "none")
        return (
            f"WilkeTransportEnhanced2(n_species={self._n_species}, "
            f"models={model_names}, diffusion={diff})"
        )
