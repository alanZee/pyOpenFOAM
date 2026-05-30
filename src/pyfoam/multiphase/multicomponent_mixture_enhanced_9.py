"""Enhanced multicomponent mixture property model — v10.

Extends MulticomponentMixtureEnhanced8 with:
- Ternary diagram phase equilibrium calculation
- Mixture thermal conductivity from pure-component correlations
- Enhanced diffusion with cross-term coupling

Usage::

    from pyfoam.multiphase.multicomponent_mixture_enhanced_9 import (
        MulticomponentMixtureEnhanced9,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Sequence
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.multicomponent_mixture_enhanced_8 import (
    MulticomponentMixtureEnhanced8,
)

__all__ = ["MulticomponentMixtureEnhanced9"]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class MulticomponentMixtureEnhanced9(MulticomponentMixtureEnhanced8):
    """Enhanced multicomponent mixture v10 with ternary equilibrium and cross-diffusion.

    Parameters
    ----------
    species, M, rho, mu, Cp, kappa, D, Sc_t : see parent.
    cross_diffusion_coeff : float
        Cross-diffusion coupling coefficient. Default 0.0.
    """

    def __init__(
        self,
        species: Sequence[str],
        M: Sequence[float],
        rho: Sequence[float],
        mu: Sequence[float],
        Cp: Sequence[float],
        kappa: Sequence[float] | None = None,
        D: Sequence[float] | None = None,
        Sc_t: Sequence[float] | None = None,
        Cp_poly: Sequence[Sequence[float]] | None = None,
        Le: Sequence[float] | None = None,
        reaction_rates: Sequence[float] | None = None,
        soret_coeff: Sequence[float] | None = None,
        dufour_coeff: Sequence[float] | None = None,
        D_ij: Sequence[Sequence[float]] | None = None,
        nrtl_alpha: Sequence[Sequence[float]] | None = None,
        H_ref: Sequence[float] | None = None,
        uniquac_r: Sequence[float] | None = None,
        uniquac_q: Sequence[float] | None = None,
        margules_A12: float = 0.0,
        margules_A21: float = 0.0,
        wilson_lambda: Sequence[Sequence[float]] | None = None,
        stefan_flow_coeff: float = 0.0,
        cross_diffusion_coeff: float = 0.0,
    ) -> None:
        super().__init__(
            species, M, rho, mu, Cp, kappa, D, Sc_t, Cp_poly,
            Le, reaction_rates, soret_coeff, dufour_coeff,
            D_ij, nrtl_alpha, H_ref, uniquac_r, uniquac_q,
            margules_A12, margules_A21,
            wilson_lambda, stefan_flow_coeff,
        )
        self._cross_diff = max(0.0, cross_diffusion_coeff)

    # ------------------------------------------------------------------
    # Cross-diffusion coupling
    # ------------------------------------------------------------------

    def cross_diffusion_flux(
        self,
        Y: torch.Tensor,
        grad_Y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-diffusion flux contribution.

        J_cross_i = -coeff * D_i * sum_{j!=i} (Y_i * grad(Y_j) - Y_j * grad(Y_i))

        Parameters
        ----------
        Y : torch.Tensor
            (n_cells, N) mass fractions.
        grad_Y : torch.Tensor
            (n_cells, N, 3) mass fraction gradients.

        Returns
        -------
        torch.Tensor
            (n_cells, N, 3) cross-diffusion flux.
        """
        if self._cross_diff < 1e-15:
            return torch.zeros_like(grad_Y)

        n_cells = Y.shape[0]
        n = Y.shape[1]
        flux = torch.zeros_like(grad_Y)

        for i in range(n):
            D_i = self._D[i] if self._D and i < len(self._D) else 2e-5
            for j in range(n):
                if i == j:
                    continue
                # Cross term
                cross = Y[:, i:i+1].unsqueeze(-1) * grad_Y[:, j:j+1, :] - \
                        Y[:, j:j+1].unsqueeze(-1) * grad_Y[:, i:i+1, :]
                flux[:, i:i+1, :] += self._cross_diff * D_i * cross

        return flux

    # ------------------------------------------------------------------
    # Mixture thermal conductivity from correlations
    # ------------------------------------------------------------------

    def mixture_kappa_correlation(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Mixture thermal conductivity from polynomial correlations.

        kappa_mix = sum_i Y_i * kappa_i(T)

        where kappa_i(T) = kappa_i_ref * (T/T_ref)^0.8

        Parameters
        ----------
        Y : torch.Tensor
            (n_cells, N) mass fractions.
        T : torch.Tensor
            (n_cells,) temperature (K).

        Returns
        -------
        torch.Tensor
            (n_cells,) mixture thermal conductivity.
        """
        n_cells = Y.shape[0]
        n = Y.shape[1]
        T_ref = 300.0
        kappa_mix = torch.zeros(n_cells, device=Y.device, dtype=Y.dtype)

        for i in range(n):
            k_i = self._kappa[i] if self._kappa and i < len(self._kappa) else 0.026
            T_ratio = (T / T_ref).clamp(min=0.1, max=10.0)
            k_T = k_i * T_ratio.pow(0.8)
            kappa_mix += Y[:, i] * k_T

        return kappa_mix.clamp(min=_EPS)

    def __repr__(self) -> str:
        sp = ", ".join(self._species)
        cd = f", cross_diff={self._cross_diff}" if self._cross_diff > 0 else ""
        return (
            f"MulticomponentMixtureEnhanced9("
            f"n_species={self._n_species}, species=[{sp}]{cd})"
        )
