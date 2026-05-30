"""Enhanced multicomponent mixture property model — v8.

Extends MulticomponentMixtureEnhanced6 with:

- **Margules活度系数模型**: Margules activity coefficient model for binary mixtures
- **反应进度追踪**: reaction progress tracking with conversion monitoring
- **混合焓计算**: mixing enthalpy computation for non-ideal solutions

Usage::

    from pyfoam.multiphase.multicomponent_mixture_enhanced_7 import (
        MulticomponentMixtureEnhanced7,
    )

    mix = MulticomponentMixtureEnhanced7(
        species=["N2", "O2", "H2O"],
        M=[28.014e-3, 32.0e-3, 18.015e-3],
        rho=[1.165, 1.331, 0.804],
        mu=[1.76e-5, 2.04e-5, 0.96e-5],
        Cp=[1040.0, 919.0, 2080.0],
        D=[2.1e-5, 2.1e-5, 2.5e-5],
    )
"""

from __future__ import annotations
import logging
from typing import Sequence
import torch
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.multicomponent_mixture_enhanced_6 import (
    MulticomponentMixtureEnhanced6,
)

__all__ = ["MulticomponentMixtureEnhanced7"]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class MulticomponentMixtureEnhanced7(MulticomponentMixtureEnhanced6):
    """Enhanced multicomponent mixture v8 with Margules activity, reaction progress,
    and mixing enthalpy.

    Extends v7 with:
    - Margules activity coefficient model for binary non-ideal mixtures
    - Reaction progress tracking with conversion monitoring
    - Mixing enthalpy for non-ideal solution effects

    Parameters
    ----------
    species, M, rho, mu, Cp, kappa, D, Sc_t, Cp_poly, Le : see parent.
    reaction_rates, soret_coeff, dufour_coeff, D_ij : see parent.
    nrtl_alpha, H_ref, uniquac_r, uniquac_q : see parent.
    margules_A12 : float
        Margules binary interaction parameter A12 (J/mol). Default 0.
    margules_A21 : float
        Margules binary interaction parameter A21 (J/mol). Default 0.
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
    ) -> None:
        super().__init__(
            species, M, rho, mu, Cp, kappa, D, Sc_t, Cp_poly,
            Le, reaction_rates, soret_coeff, dufour_coeff,
            D_ij, nrtl_alpha, H_ref, uniquac_r, uniquac_q,
        )
        self._margules_A12 = margules_A12
        self._margules_A21 = margules_A21
        self._reaction_progress: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Margules activity coefficients (binary)
    # ------------------------------------------------------------------

    def margules_activity_coefficients(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Margules activity coefficients for binary mixture.

        ln(gamma_1) = x_2^2 * [A12 + 2*(A21 - A12)*x_1]
        ln(gamma_2) = x_1^2 * [A21 + 2*(A12 - A21)*x_2]

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
        R_univ = 8.314462618
        X = self.mass_to_mole(Y)
        n_cells = X.shape[0]
        device = X.device
        dtype = X.dtype
        n_species = min(X.shape[1], 2)  # Margules is binary

        gamma = torch.ones_like(X)
        if n_species < 2:
            return gamma

        x1 = X[:, 0].clamp(_EPS, 1.0 - _EPS)
        x2 = (1.0 - x1).clamp(_EPS, 1.0 - _EPS)
        T_safe = T.clamp(min=1.0)

        A12_T = self._margules_A12 / (R_univ * T_safe)
        A21_T = self._margules_A21 / (R_univ * T_safe)

        ln_gamma1 = x2.pow(2) * (A12_T + 2.0 * (A21_T - A12_T) * x1)
        ln_gamma2 = x1.pow(2) * (A21_T + 2.0 * (A12_T - A21_T) * x2)

        gamma[:, 0] = ln_gamma1.exp().clamp(min=0.1, max=10.0)
        gamma[:, 1] = ln_gamma2.exp().clamp(min=0.1, max=10.0)

        return gamma

    # ------------------------------------------------------------------
    # Reaction progress tracking
    # ------------------------------------------------------------------

    def track_reaction_progress(
        self,
        Y: torch.Tensor,
        Y_initial: torch.Tensor,
        species_idx: int,
    ) -> dict[str, float]:
        """Compute reaction conversion for a species.

        Parameters
        ----------
        Y : torch.Tensor
            Current mass fractions.
        Y_initial : torch.Tensor
            Initial mass fractions.
        species_idx : int
            Target species index.

        Returns
        -------
        dict
            'conversion': fraction converted (0 to 1),
            'Y_current': current mass fraction,
            'Y_initial': initial mass fraction.
        """
        Y_curr = float(Y[:, species_idx].mean().item())
        Y_init = float(Y_initial[:, species_idx].mean().item())
        if abs(Y_init) < _EPS:
            conversion = 0.0
        else:
            conversion = max(0.0, min(1.0, (Y_init - Y_curr) / Y_init))

        result = {
            "conversion": conversion,
            "Y_current": Y_curr,
            "Y_initial": Y_init,
        }
        self._reaction_progress.append(result)
        return result

    # ------------------------------------------------------------------
    # Mixing enthalpy
    # ------------------------------------------------------------------

    def mixing_enthalpy(self, Y: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Compute mixing enthalpy for non-ideal solution.

        H_mix = sum_i x_i * H_i + H_excess

        where H_excess ~ R * T^2 * sum_i x_i * d(ln gamma_i)/dT

        Simplified: H_excess ~ R * T * sum(x_i * ln(gamma_i))

        Parameters
        ----------
        Y : torch.Tensor
            (n_cells, N) mass fractions.
        T : torch.Tensor
            (n_cells,) temperature.

        Returns
        -------
        torch.Tensor
            (n_cells,) mixing enthalpy (J/kg).
        """
        R_univ = 8.314462618
        gamma = self.margules_activity_coefficients(Y, T)
        X = self.mass_to_mole(Y)
        ln_gamma = gamma.clamp(min=_EPS).log()
        H_excess = R_univ * T.clamp(min=1.0) * (X * ln_gamma).sum(dim=1)
        return H_excess

    def __repr__(self) -> str:
        sp = ", ".join(self._species)
        return (
            f"MulticomponentMixtureEnhanced7("
            f"n_species={self._n_species}, species=[{sp}])"
        )
