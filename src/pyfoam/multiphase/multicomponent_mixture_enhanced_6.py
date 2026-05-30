"""Enhanced multicomponent mixture property model — v7.

Extends MulticomponentMixtureEnhanced5 with:

- **UNIQUAC活度系数模型**: UNIQUAC (UNIversal QUAsiChemical) activity model
- **扩散限制反应速率**: diffusion-limited reaction rate computation
- **混合规则验证**: validate mixture properties against experimental data

Usage::

    from pyfoam.multiphase.multicomponent_mixture_enhanced_6 import (
        MulticomponentMixtureEnhanced6,
    )

    mix = MulticomponentMixtureEnhanced6(
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
from pyfoam.multiphase.multicomponent_mixture_enhanced_5 import (
    MulticomponentMixtureEnhanced5,
)

__all__ = ["MulticomponentMixtureEnhanced6"]
logger = logging.getLogger(__name__)
_EPS = 1e-30


class MulticomponentMixtureEnhanced6(MulticomponentMixtureEnhanced5):
    """Enhanced multicomponent mixture v7 with UNIQUAC, diffusion-limited
    reactions, and mixture validation.

    Extends v6 with:
    - UNIQUAC activity coefficient model for non-ideal liquid mixtures
    - Diffusion-limited reaction rate estimation
    - Mixture property validation against reference data

    Parameters
    ----------
    species, M, rho, mu, Cp, kappa, D, Sc_t, Cp_poly, Le : see parent
    reaction_rates, soret_coeff, dufour_coeff, D_ij : see parent
    nrtl_alpha, H_ref : see parent
    uniquac_r : sequence of float, optional
        UNIQUAC volume parameter per species. Default None.
    uniquac_q : sequence of float, optional
        UNIQUAC surface parameter per species. Default None.
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
    ) -> None:
        super().__init__(
            species, M, rho, mu, Cp, kappa, D, Sc_t, Cp_poly,
            Le, reaction_rates, soret_coeff, dufour_coeff,
            D_ij, nrtl_alpha, H_ref,
        )
        self._uniquac_r = list(uniquac_r) if uniquac_r is not None else [1.0] * self._n_species
        self._uniquac_q = list(uniquac_q) if uniquac_q is not None else [1.0] * self._n_species

    # ------------------------------------------------------------------
    # UNIQUAC activity coefficients
    # ------------------------------------------------------------------

    def uniquac_activity_coefficients(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute UNIQUAC activity coefficients.

        Simplified UNIQUAC:
            ln(gamma_i) = ln(phi_i/x_i) + (z/2) * q_i * ln(theta_i/phi_i)
                        + l_i - phi_i/x_i * sum(x_j * l_j)

        where phi_i = r_i * x_i / sum(r_j * x_j),
              theta_i = q_i * x_i / sum(q_j * x_j).

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
        device = Y.device
        dtype = Y.dtype
        X = self.mass_to_mole(Y)
        n_cells = X.shape[0]

        r = torch.tensor(self._uniquac_r, device=device, dtype=dtype)
        q = torch.tensor(self._uniquac_q, device=device, dtype=dtype)

        # Volume and surface fractions
        rx = r.unsqueeze(0) * X  # (n_cells, N)
        qx = q.unsqueeze(0) * X
        rx_sum = rx.sum(dim=1, keepdim=True).clamp(min=_EPS)
        qx_sum = qx.sum(dim=1, keepdim=True).clamp(min=_EPS)
        phi = rx / rx_sum  # (n_cells, N)
        theta = qx / qx_sum

        z = 10.0  # Coordination number
        l_i = (z / 2.0) * (r - q) - (r - 1.0)  # (N,)
        l_i = l_i.unsqueeze(0).expand(n_cells, -1)

        X_safe = X.clamp(min=_EPS)
        ln_gamma = (
            torch.log(phi / X_safe)
            + (z / 2.0) * q.unsqueeze(0) * torch.log(theta / phi).clamp(min=-20)
            + l_i - phi / X_safe * (X * l_i).sum(dim=1, keepdim=True)
        )

        gamma = ln_gamma.exp().clamp(min=0.1, max=10.0)
        return gamma

    # ------------------------------------------------------------------
    # Diffusion-limited reaction rate
    # ------------------------------------------------------------------

    def diffusion_limited_rate(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        species_idx: int,
        activation_energy: float = 50e3,
    ) -> torch.Tensor:
        """Compute diffusion-limited reaction rate.

        k_eff = D_i * exp(-Ea / (R * T))

        Parameters
        ----------
        Y : torch.Tensor
            (n_cells, N) mass fractions.
        T : torch.Tensor
            (n_cells,) temperature (K).
        species_idx : int
            Target species index.
        activation_energy : float
            Activation energy (J/mol). Default 50e3.

        Returns
        -------
        torch.Tensor
            (n_cells,) effective reaction rate.
        """
        R_univ = 8.314462618
        D_i = self._D[min(species_idx, len(self._D) - 1)]
        T_safe = T.clamp(min=1.0)
        return D_i * torch.exp(-activation_energy / (R_univ * T_safe))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_mixture_viscosity(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        mu_reference: float,
    ) -> dict[str, float]:
        """Compare computed mixture viscosity against reference."""
        mu_computed = self.mixture_viscosity(Y)
        rel_err = abs(mu_computed.mean().item() - mu_reference) / max(abs(mu_reference), _EPS)
        return {
            "mu_computed": mu_computed.mean().item(),
            "mu_reference": mu_reference,
            "relative_error": rel_err,
        }

    def __repr__(self) -> str:
        sp = ", ".join(self._species)
        return (
            f"MulticomponentMixtureEnhanced6("
            f"n_species={self._n_species}, species=[{sp}])"
        )
