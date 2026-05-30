"""Enhanced multicomponent mixture property model — v9.

Extends MulticomponentMixtureEnhanced7 with:

- **Wilson 活度系数模型**: Wilson activity coefficient model for multicomponent mixtures
- **Wassiljewa 导热系数**: Wassiljewa mixing rule for thermal conductivity
- **Stefan 流修正**: Stefan flow correction for mass transfer at interfaces

Usage::

    from pyfoam.multiphase.multicomponent_mixture_enhanced_8 import (
        MulticomponentMixtureEnhanced8,
    )
"""

from __future__ import annotations
import logging
import math
from typing import Sequence
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.multicomponent_mixture_enhanced_7 import (
    MulticomponentMixtureEnhanced7,
)

__all__ = ["MulticomponentMixtureEnhanced8"]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class MulticomponentMixtureEnhanced8(MulticomponentMixtureEnhanced7):
    """Enhanced multicomponent mixture v9 with Wilson activity, Wassiljewa conductivity,
    and Stefan flow correction.

    Extends v8 with:
    - Wilson activity coefficient model for multicomponent mixtures
    - Wassiljewa mixing rule for thermal conductivity
    - Stefan flow correction for mass transfer at interfaces

    Parameters
    ----------
    species, M, rho, mu, Cp, kappa, D, Sc_t, Cp_poly, Le : see parent.
    reaction_rates, soret_coeff, dufour_coeff, D_ij : see parent.
    nrtl_alpha, H_ref, uniquac_r, uniquac_q : see parent.
    margules_A12, margules_A21 : see parent.
    wilson_lambda : list of list of float or None
        Wilson binary interaction parameters (dimensionless). Default None.
    stefan_flow_coeff : float
        Stefan flow correction coefficient. Default 0.0.
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
    ) -> None:
        super().__init__(
            species, M, rho, mu, Cp, kappa, D, Sc_t, Cp_poly,
            Le, reaction_rates, soret_coeff, dufour_coeff,
            D_ij, nrtl_alpha, H_ref, uniquac_r, uniquac_q,
            margules_A12, margules_A21,
        )
        self._wilson_lambda = wilson_lambda
        self._stefan_coeff = max(0.0, stefan_flow_coeff)

    # ------------------------------------------------------------------
    # Wilson activity coefficients
    # ------------------------------------------------------------------

    def wilson_activity_coefficients(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Wilson activity coefficients for multicomponent mixture.

        ln(gamma_i) = 1 - ln(sum_j x_j * Lambda_ij) - sum_j (x_j * Lambda_ji / sum_k x_k * Lambda_kj)

        Parameters
        ----------
        Y : torch.Tensor
            (n_cells, N) mass fractions.
        T : torch.Tensor
            (n_cells,) temperature (K).

        Returns
        -------
        torch.Tensor
            (n_cells, N) activity coefficients.
        """
        X = self.mass_to_mole(Y)
        n_cells = X.shape[0]
        n = X.shape[1]

        gamma = torch.ones_like(X)

        if self._wilson_lambda is None or n < 2:
            return gamma

        for i in range(n):
            # Compute sum_j x_j * Lambda_ij
            sum_xL = torch.zeros(n_cells, device=X.device, dtype=X.dtype)
            for j in range(n):
                L_ij = self._wilson_lambda[i][j] if i < len(self._wilson_lambda) and j < len(self._wilson_lambda[i]) else 1.0
                sum_xL = sum_xL + X[:, j] * L_ij

            # Compute the second sum
            sum_term = torch.zeros(n_cells, device=X.device, dtype=X.dtype)
            for j in range(n):
                L_ji = self._wilson_lambda[j][i] if j < len(self._wilson_lambda) and i < len(self._wilson_lambda[j]) else 1.0
                sum_xL_j = torch.zeros(n_cells, device=X.device, dtype=X.dtype)
                for k in range(n):
                    L_kj = self._wilson_lambda[k][j] if k < len(self._wilson_lambda) and j < len(self._wilson_lambda[k]) else 1.0
                    sum_xL_j = sum_xL_j + X[:, k] * L_kj
                sum_term = sum_term + X[:, j] * L_ji / sum_xL_j.clamp(min=_EPS)

            ln_gamma = 1.0 - torch.log(sum_xL.clamp(min=_EPS)) - sum_term
            gamma[:, i] = ln_gamma.exp().clamp(min=0.1, max=10.0)

        return gamma

    # ------------------------------------------------------------------
    # Wassiljewa conductivity
    # ------------------------------------------------------------------

    def wassiljewa_conductivity(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture thermal conductivity using Wassiljewa mixing rule.

        kappa_mix = sum_i (x_i * kappa_i) / sum_j (x_j * A_ij)

        where A_ij ~ 0.25 * (1 + (mu_i/mu_j)^0.5 * (M_j/M_i)^0.25)^2

        Parameters
        ----------
        Y : torch.Tensor
            (n_cells, N) mass fractions.
        T : torch.Tensor
            (n_cells,) temperature (K).

        Returns
        -------
        torch.Tensor
            (n_cells,) mixture thermal conductivity (W/(m*K)).
        """
        X = self.mass_to_mole(Y)
        n_cells = X.shape[0]
        n = X.shape[1]

        if self._kappa is None or n == 0:
            return torch.full((n_cells,), 0.026, device=Y.device, dtype=Y.dtype)

        kappa_mix = torch.zeros(n_cells, device=Y.device, dtype=Y.dtype)
        for i in range(n):
            k_i = self._kappa[i] if i < len(self._kappa) else 0.026
            denom = torch.zeros(n_cells, device=Y.device, dtype=Y.dtype)
            for j in range(n):
                # A_ij approximation
                mu_i = self._mu[i] if i < len(self._mu) else 1.8e-5
                mu_j = self._mu[j] if j < len(self._mu) else 1.8e-5
                M_j = self._M[j] if j < len(self._M) else 0.029
                M_i = self._M[i] if i < len(self._M) else 0.029
                A_ij = 0.25 * (1.0 + math.sqrt(mu_i / max(mu_j, _EPS)) * (M_j / max(M_i, _EPS)) ** 0.25) ** 2
                denom = denom + X[:, j] * A_ij
            kappa_mix = kappa_mix + X[:, i] * k_i / denom.clamp(min=_EPS)

        return kappa_mix.clamp(min=_EPS)

    # ------------------------------------------------------------------
    # Stefan flow correction
    # ------------------------------------------------------------------

    def stefan_flow_correction(
        self,
        Y: torch.Tensor,
        Y_interface: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Stefan flow velocity correction at interface.

        v_stefan = coeff * sum_i (D_i * grad(Y_i) / (1 - Y_i))

        Parameters
        ----------
        Y : torch.Tensor
            (n_cells, N) bulk mass fractions.
        Y_interface : torch.Tensor
            (n_cells, N) interface mass fractions.

        Returns
        -------
        torch.Tensor
            (n_cells,) Stefan flow velocity correction (m/s).
        """
        if self._stefan_coeff < 1e-15:
            return torch.zeros(Y.shape[0], device=Y.device, dtype=Y.dtype)

        n = Y.shape[1]
        v_stefan = torch.zeros(Y.shape[0], device=Y.device, dtype=Y.dtype)
        for i in range(n):
            D_i = self._D[i] if self._D and i < len(self._D) else 2e-5
            dY = (Y_interface[:, i] - Y[:, i]).abs()
            Y_mean = ((Y[:, i] + Y_interface[:, i]) / 2.0).clamp(_EPS, 1.0 - _EPS)
            v_stefan = v_stefan + D_i * dY / (1.0 - Y_mean)

        return self._stefan_coeff * v_stefan

    def __repr__(self) -> str:
        sp = ", ".join(self._species)
        return (
            f"MulticomponentMixtureEnhanced8("
            f"n_species={self._n_species}, species=[{sp}])"
        )
