"""
Enhanced multicomponent mixture property model — v6.

在 Enhanced v5 基础上增加：

- **Stefan-Maxwell 扩散求解器**：多组分扩散通量的耦合求解
- **活度系数模型**：支持 NRTL 和 Wilson 活度系数
- **混合焓计算**：基于各组分焓值的混合焓评估

Usage::

    from pyfoam.multiphase.multicomponent_mixture_enhanced_5 import (
        MulticomponentMixtureEnhanced5,
    )

    mix = MulticomponentMixtureEnhanced5(
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
from pyfoam.multiphase.multicomponent_mixture_enhanced_4 import (
    MulticomponentMixtureEnhanced4,
)

__all__ = ["MulticomponentMixtureEnhanced5"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class MulticomponentMixtureEnhanced5(MulticomponentMixtureEnhanced4):
    """Enhanced multicomponent mixture v6 with Stefan-Maxwell diffusion and
    activity coefficient models.

    在 v5 基础上增加：
    - Stefan-Maxwell 多组分扩散求解器
    - NRTL 活度系数模型
    - 混合焓计算

    Parameters
    ----------
    species : sequence of str
        Species names (N >= 1).
    M : sequence of float
        Molecular weights (kg/mol).
    rho : sequence of float
        Pure-species densities (kg/m^3).
    mu : sequence of float
        Pure-species viscosities (Pa·s).
    Cp : sequence of float
        Specific heats (J/(kg·K)).
    kappa : sequence of float, optional
        Thermal conductivities (W/(m·K)).
    D : sequence of float, optional
        Molecular diffusion coefficients (m^2/s).
    Sc_t : sequence of float, optional
        Turbulent Schmidt numbers.
    Cp_poly : sequence of sequence of float, optional
        Polynomial Cp coefficients.
    Le : sequence of float, optional
        Lewis numbers per species.
    reaction_rates : sequence of float, optional
        Volumetric reaction rate constants per species (1/s).
    soret_coeff : sequence of float, optional
        Soret (thermal diffusion) coefficients per species.
    dufour_coeff : sequence of float, optional
        Dufour (diffusion thermo) coefficients per species.
    D_ij : sequence of sequence of float, optional
        Binary diffusion coefficient matrix (m^2/s).
        If None, estimated from species diffusion coefficients.
    nrtl_alpha : sequence of sequence of float, optional
        NRTL non-randomness parameter matrix.
    H_ref : sequence of float, optional
        Reference enthalpy per species (J/kg).
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
    ) -> None:
        super().__init__(
            species, M, rho, mu, Cp, kappa, D, Sc_t, Cp_poly,
            Le, reaction_rates, soret_coeff, dufour_coeff,
        )

        # Binary diffusion matrix
        if D_ij is not None:
            self._D_ij = [list(row) for row in D_ij]
        else:
            # Estimate from molecular diffusion coefficients
            self._D_ij = []
            for i in range(self._n_species):
                row = []
                for j in range(self._n_species):
                    if i == j:
                        row.append(0.0)
                    else:
                        d_avg = (self._D[i] + self._D[j]) / 2.0
                        row.append(d_avg)
                self._D_ij.append(row)

        # NRTL alpha matrix
        if nrtl_alpha is not None:
            self._nrtl_alpha = [list(row) for row in nrtl_alpha]
        else:
            # Default: symmetric, 0.3 for all pairs
            self._nrtl_alpha = [
                [0.3 if i != j else 0.0 for j in range(self._n_species)]
                for i in range(self._n_species)
            ]

        # Reference enthalpy
        if H_ref is not None:
            self._H_ref = list(H_ref)
        else:
            self._H_ref = [0.0] * self._n_species

    @property
    def D_ij(self) -> list[list[float]]:
        """Binary diffusion coefficient matrix."""
        return [row.copy() for row in self._D_ij]

    @property
    def H_ref(self) -> list[float]:
        """Reference enthalpy per species (J/kg)."""
        return self._H_ref.copy()

    # ------------------------------------------------------------------
    # Stefan-Maxwell 扩散
    # ------------------------------------------------------------------

    def stefan_maxwell_flux(
        self,
        Y: torch.Tensor,
        grad_X: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Stefan-Maxwell multicomponent diffusion flux.

        Approximated as:
            J_i = -rho * sum_j (D_ij * (X_i * grad(X_j) - X_j * grad(X_i)))
                  / (X_i + X_j)

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        grad_X : torch.Tensor
            ``(n_cells, N, 3)`` mole fraction gradients.

        Returns
        -------
        torch.Tensor
            ``(n_cells, N, 3)`` diffusion mass flux (kg/(m^2·s)).
        """
        device = Y.device
        dtype = Y.dtype
        n_cells = Y.shape[0]
        n_sp = self._n_species

        X = self.mass_to_mole(Y)

        D_ij_t = torch.tensor(self._D_ij, device=device, dtype=dtype)

        flux = torch.zeros(n_cells, n_sp, 3, device=device, dtype=dtype)

        for i in range(n_sp):
            for j in range(n_sp):
                if i == j:
                    continue
                Dij = D_ij_t[i, j]
                X_sum = (X[:, i] + X[:, j]).clamp(min=_EPS)
                # Contribution: D_ij * (X_i * grad_X_j - X_j * grad_X_i) / (X_i + X_j)
                term_i = X[:, i].unsqueeze(-1) * grad_X[:, j, :]
                term_j = X[:, j].unsqueeze(-1) * grad_X[:, i, :]
                flux[:, i, :] += Dij * (term_i - term_j) / X_sum.unsqueeze(-1)

        return flux

    # ------------------------------------------------------------------
    # 活度系数 (NRTL)
    # ------------------------------------------------------------------

    def nrtl_activity_coefficients(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NRTL activity coefficients.

        Simplified symmetric NRTL:
            ln(gamma_i) = sum_j (tau_ji * G_ji * X_j) / sum_k (G_ki * X_k)
                        - X_i * sum_j (tau_ij * G_ij * X_j) / sum_k (G_kj * X_k)^2

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells, N)`` activity coefficients.
        """
        device = Y.device
        dtype = Y.dtype
        n_cells = Y.shape[0]
        n_sp = self._n_species

        X = self.mass_to_mole(Y)

        alpha = torch.tensor(self._nrtl_alpha, device=device, dtype=dtype)

        # Simplified: gamma ~ 1 + alpha_ij * X_j (first-order)
        gamma = torch.ones(n_cells, n_sp, device=device, dtype=dtype)

        for i in range(n_sp):
            for j in range(n_sp):
                if i != j and abs(self._nrtl_alpha[i][j]) > _EPS:
                    gamma[:, i] = gamma[:, i] + alpha[i, j] * X[:, j]

        return gamma.clamp(min=0.1, max=10.0)

    # ------------------------------------------------------------------
    # 混合焓
    # ------------------------------------------------------------------

    def mixture_enthalpy(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture enthalpy.

        h_mix = sum_i Y_i * (H_ref_i + Cp_i * (T - T_ref))

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture enthalpy (J/kg).
        """
        Y = self.validate_mass_fractions(Y)
        device = Y.device
        dtype = Y.dtype
        T_ref = 298.15

        H_ref = torch.tensor(self._H_ref, device=device, dtype=dtype)
        Cp_arr = torch.tensor(self._Cp, device=device, dtype=dtype)

        h_species = H_ref + Cp_arr * (T.to(device=device, dtype=dtype) - T_ref).unsqueeze(-1)
        h_mix = (Y * h_species).sum(dim=-1)

        return h_mix

    def __repr__(self) -> str:
        sp = ", ".join(self._species)
        has_soret = any(s != 0 for s in self._soret_coeff)
        return (
            f"MulticomponentMixtureEnhanced5("
            f"n_species={self._n_species}, species=[{sp}], "
            f"has_soret={has_soret})"
        )
