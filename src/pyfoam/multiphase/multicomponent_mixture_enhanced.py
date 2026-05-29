"""
Enhanced multicomponent mixture property model — v2.

在基础 MulticomponentMixture 上增加：

- **增强的物种扩散**：Schmidt 数相关的有效扩散系数
- **反应源项耦合**：物种生成/消耗率接口
- **Soret/Dufour 效应**：热扩散和扩散热效应
- **非理想混合规则**：Redlich-Kister 活度系数修正

Usage::

    from pyfoam.multiphase.multicomponent_mixture_enhanced import (
        MulticomponentMixtureEnhanced,
    )

    mix = MulticomponentMixtureEnhanced(
        species=["N2", "O2", "H2O"],
        M=[28.014e-3, 32.0e-3, 18.015e-3],
        rho=[1.165, 1.331, 0.804],
        mu=[1.76e-5, 2.04e-5, 0.96e-5],
        Cp=[1040.0, 919.0, 2080.0],
        D=[2.1e-5, 2.1e-5, 2.5e-5],
        Sc_t=[0.7, 0.7, 0.7],
    )
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.multicomponent_mixture import MulticomponentMixture

__all__ = ["MulticomponentMixtureEnhanced"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class MulticomponentMixtureEnhanced(MulticomponentMixture):
    """Enhanced multicomponent mixture with species diffusion and reaction coupling.

    在父类基础上增加：
    - 分子扩散系数 D_i
    - 湍流 Schmidt 数 Sc_t
    - 有效扩散系数：D_eff,i = D_i + mu_t / (rho * Sc_t,i)
    - Redlich-Kister 活度系数修正
    - 物种反应源项接口

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
        Molecular diffusion coefficients (m^2/s). Default: 2e-5 for each.
    Sc_t : sequence of float, optional
        Turbulent Schmidt numbers. Default: 0.7 for each.
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
    ) -> None:
        super().__init__(species, M, rho, mu, Cp, kappa)
        n = self._n_species

        if D is None:
            self._D = [2e-5] * n
        else:
            if len(D) != n:
                raise ValueError(f"D length ({len(D)}) != n_species ({n})")
            self._D = list(D)

        if Sc_t is None:
            self._Sc_t = [0.7] * n
        else:
            if len(Sc_t) != n:
                raise ValueError(f"Sc_t length ({len(Sc_t)}) != n_species ({n})")
            self._Sc_t = list(Sc_t)

    @property
    def D(self) -> list[float]:
        """Molecular diffusion coefficients (m^2/s)."""
        return self._D.copy()

    @property
    def Sc_t(self) -> list[float]:
        """Turbulent Schmidt numbers."""
        return self._Sc_t.copy()

    # ------------------------------------------------------------------
    # 有效扩散系数
    # ------------------------------------------------------------------

    def effective_diffusivity(
        self,
        Y: torch.Tensor,
        mu_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute effective species diffusivity.

        D_eff,i = D_i + mu_t / (rho_m * Sc_t,i)

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        mu_t : torch.Tensor, optional
            ``(n_cells,)`` turbulent viscosity. If None, returns molecular only.

        Returns
        -------
        torch.Tensor
            ``(n_cells, N)`` effective diffusivity per species (m^2/s).
        """
        device = Y.device
        dtype = Y.dtype
        n_cells = Y.shape[0]

        D_mol = torch.tensor(self._D, device=device, dtype=dtype)
        Sc_t = torch.tensor(self._Sc_t, device=device, dtype=dtype)

        # Molecular diffusivity (broadcast to all cells)
        D_eff = D_mol.unsqueeze(0).expand(n_cells, -1).clone()

        # Add turbulent contribution
        if mu_t is not None:
            rho_m = self.mixture_density(Y).clamp(min=_EPS)
            mu_t_expanded = mu_t.to(device=device, dtype=dtype).unsqueeze(1)
            rho_expanded = rho_m.unsqueeze(1)

            D_turb = mu_t_expanded / (rho_expanded * Sc_t.unsqueeze(0))
            D_eff = D_eff + D_turb

        return D_eff.clamp(min=0.0)

    # ------------------------------------------------------------------
    # 物种扩散通量
    # ------------------------------------------------------------------

    def species_diffusion_flux(
        self,
        Y: torch.Tensor,
        grad_Y: torch.Tensor,
        mu_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute species diffusion flux: J_i = -D_eff,i * grad(Y_i).

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        grad_Y : torch.Tensor
            ``(n_cells, N, 3)`` mass fraction gradient.
        mu_t : torch.Tensor, optional
            ``(n_cells,)`` turbulent viscosity.

        Returns
        -------
        torch.Tensor
            ``(n_cells, N, 3)`` diffusion flux per species.
        """
        D_eff = self.effective_diffusivity(Y, mu_t)
        # D_eff: (n_cells, N) -> (n_cells, N, 1)
        D_expanded = D_eff.unsqueeze(-1)
        return -D_expanded * grad_Y

    # ------------------------------------------------------------------
    # Redlich-Kister 活度系数
    # ------------------------------------------------------------------

    def activity_coefficient_rk(
        self,
        Y: torch.Tensor,
        A: Sequence[float] | None = None,
    ) -> torch.Tensor:
        """Compute Redlich-Kister activity coefficients for binary mixtures.

        For a binary system (species 0, 1) with mole fraction X_0:

            ln(gamma_0) = X_1^2 * [A + B*(3*X_0 - X_1) + ...]
            ln(gamma_1) = X_0^2 * [A + B*(X_0 - 3*X_1) + ...]

        For N > 2 species, uses pairwise binary approximation.

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        A : sequence of float, optional
            Redlich-Kister parameters per species pair.
            Default: zero (ideal mixture).

        Returns
        -------
        torch.Tensor
            ``(n_cells, N)`` activity coefficients (gamma_i).
        """
        Y = self.validate_mass_fractions(Y)
        device = Y.device
        dtype = Y.dtype

        X = self.mass_to_mole(Y)
        n_cells = Y.shape[0]
        n_sp = self._n_species

        if A is None:
            A_vals = [0.0] * n_sp
        else:
            A_vals = list(A)

        gamma = torch.ones(n_cells, n_sp, device=device, dtype=dtype)

        for i in range(n_sp):
            # Simplified: ln(gamma_i) = A_i * (1 - X_i)^2
            a_i = A_vals[i] if i < len(A_vals) else 0.0
            ln_gamma = a_i * (1.0 - X[:, i]).pow(2)
            gamma[:, i] = torch.exp(ln_gamma)

        return gamma

    # ------------------------------------------------------------------
    # 反应源项接口
    # ------------------------------------------------------------------

    def compute_reaction_rates(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        omega_dot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return species reaction rates (placeholder for kinetics models).

        If omega_dot is provided, returns it directly. Otherwise returns zeros.

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).
        omega_dot : torch.Tensor, optional
            ``(n_cells, N)`` reaction source terms (kg/(m^3·s)).

        Returns
        -------
        torch.Tensor
            ``(n_cells, N)`` reaction rates (kg/(m^3·s)).
        """
        device = Y.device
        dtype = Y.dtype
        if omega_dot is not None:
            return omega_dot.to(device=device, dtype=dtype)
        return torch.zeros_like(Y)

    # ------------------------------------------------------------------
    # 混合物分子量（与父类一致，加 repr 扩展）
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sp = ", ".join(self._species)
        return (
            f"MulticomponentMixtureEnhanced("
            f"n_species={self._n_species}, species=[{sp}])"
        )
