"""
Enhanced multicomponent mixture property model — v3.

在 Enhanced v2 基础上增加：

- **逆梯度扩散修正**：counter-gradient diffusion correction
- **Maxwell-Stefan 多组分扩散**：多组分相互扩散矩阵
- **增强混合规则**：Wilke 混合粘性、多项式 Cp(T)

Usage::

    from pyfoam.multiphase.multicomponent_mixture_enhanced_2 import (
        MulticomponentMixtureEnhanced2,
    )

    mix = MulticomponentMixtureEnhanced2(
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
from pyfoam.multiphase.multicomponent_mixture_enhanced import (
    MulticomponentMixtureEnhanced,
)

__all__ = ["MulticomponentMixtureEnhanced2"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class MulticomponentMixtureEnhanced2(MulticomponentMixtureEnhanced):
    """Enhanced multicomponent mixture v3 with improved species transport.

    在 v2 基础上增加：
    - 逆梯度扩散修正 (counter-gradient)
    - Maxwell-Stefan 相互扩散矩阵
    - Wilke 混合粘性
    - 多项式 Cp(T)

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
        Polynomial Cp coefficients per species: [a0, a1, a2, ...]
        where Cp(T) = a0 + a1*T + a2*T^2 + ...
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
    ) -> None:
        super().__init__(species, M, rho, mu, Cp, kappa, D, Sc_t)

        # Polynomial Cp coefficients: [[a0, a1, a2], ...]
        if Cp_poly is not None:
            if len(Cp_poly) != self._n_species:
                raise ValueError(
                    f"Cp_poly length ({len(Cp_poly)}) != n_species ({self._n_species})"
                )
            self._Cp_poly = [list(c) for c in Cp_poly]
        else:
            # Default: constant Cp (just a0)
            self._Cp_poly = [[cp] for cp in Cp]

    @property
    def Cp_poly(self) -> list[list[float]]:
        """Polynomial Cp coefficients per species."""
        return [c.copy() for c in self._Cp_poly]

    # ------------------------------------------------------------------
    # 多项式 Cp(T)
    # ------------------------------------------------------------------

    def Cp_temperature(
        self,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute species-specific Cp as a function of temperature.

        Cp_i(T) = a0_i + a1_i * T + a2_i * T^2 + ...

        Parameters
        ----------
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells, N)`` species Cp (J/(kg·K)).
        """
        device = T.device
        dtype = T.dtype
        T_safe = T.clamp(min=1.0)
        n_cells = T.shape[0]

        Cp = torch.zeros(n_cells, self._n_species, device=device, dtype=dtype)

        for i in range(self._n_species):
            coeffs = self._Cp_poly[i]
            cp_i = torch.zeros_like(T_safe)
            for j, a_j in enumerate(coeffs):
                cp_i = cp_i + a_j * T_safe.pow(j)
            Cp[:, i] = cp_i

        return Cp.clamp(min=_EPS)

    # ------------------------------------------------------------------
    # Wilke 混合粘性
    # ------------------------------------------------------------------

    def wilke_viscosity(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture viscosity using Wilke's semi-empirical method.

        mu_m = sum_i (X_i * mu_i / sum_j (X_j * Phi_ij))

        where Phi_ij = (1 + (mu_i/mu_j)^(1/2) * (M_j/M_i)^(1/4))^2
                     / (8 * (1 + M_i/M_j))^(1/2)

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture viscosity (Pa·s).
        """
        Y = self.validate_mass_fractions(Y)
        device = Y.device
        dtype = Y.dtype
        n_cells = Y.shape[0]
        n_sp = self._n_species

        # Convert to mole fractions
        X = self.mass_to_mole(Y)

        # Pure-species viscosities (temperature-independent for now)
        mu_pure = torch.tensor(self._mu, device=device, dtype=dtype)  # (N,)

        # Molecular weights
        M_arr = torch.tensor(self._M, device=device, dtype=dtype)  # (N,)

        # Compute Phi_ij matrix (N, N)
        # For each cell, we compute a weighted sum
        # Simplified: compute the mixing rule using mole fraction weights

        mu_mix = torch.zeros(n_cells, device=device, dtype=dtype)

        for i in range(n_sp):
            # Denominator: sum_j X_j * Phi_ij
            denom = torch.zeros(n_cells, device=device, dtype=dtype)
            for j in range(n_sp):
                if i == j:
                    denom = denom + X[:, j]
                else:
                    mu_ratio = (mu_pure[i] / mu_pure[j].clamp(min=_EPS)).sqrt()
                    M_ratio = (M_arr[j] / M_arr[i].clamp(min=_EPS)).pow(0.25)
                    phi_ij = (1.0 + mu_ratio * M_ratio).pow(2)
                    phi_ij = phi_ij / (8.0 * (1.0 + M_arr[i] / M_arr[j].clamp(min=_EPS))).sqrt()
                    denom = denom + X[:, j] * phi_ij

            mu_mix = mu_mix + X[:, i] * mu_pure[i] / denom.clamp(min=_EPS)

        return mu_mix.clamp(min=_EPS)

    # ------------------------------------------------------------------
    # Maxwell-Stefan 扩散矩阵
    # ------------------------------------------------------------------

    def maxwell_stefan_diffusivity(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Maxwell-Stefan binary diffusion coefficients.

        Approximate binary diffusivity:

            D_ij = D_ref * (T/T_ref)^1.75 * (p_ref/p)

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).
        p : torch.Tensor
            ``(n_cells,)`` pressure (Pa).

        Returns
        -------
        torch.Tensor
            ``(n_cells, N, N)`` binary diffusion coefficients (m^2/s).
        """
        device = Y.device
        dtype = Y.dtype
        n_cells = Y.shape[0]
        n_sp = self._n_species

        T_ref = 300.0
        p_ref = 101325.0

        T_safe = T.clamp(min=1.0)
        p_safe = p.clamp(min=1.0)

        D_ref = torch.tensor(self._D, device=device, dtype=dtype)  # (N,)

        # Temperature and pressure correction
        correction = (T_safe / T_ref).pow(1.75) * (p_ref / p_safe)
        correction = correction.clamp(min=_EPS)  # (n_cells,)

        # Build (N, N) matrix using geometric mean of species diffusivities
        D_matrix = torch.zeros(
            n_cells, n_sp, n_sp, device=device, dtype=dtype,
        )
        for i in range(n_sp):
            for j in range(n_sp):
                if i != j:
                    D_ij_ref = (D_ref[i] * D_ref[j]).sqrt()
                    D_matrix[:, i, j] = D_ij_ref * correction

        return D_matrix

    # ------------------------------------------------------------------
    # 逆梯度扩散修正
    # ------------------------------------------------------------------

    def counter_gradient_correction(
        self,
        grad_Y: torch.Tensor,
        D_eff: torch.Tensor,
    ) -> torch.Tensor:
        """Compute counter-gradient diffusion correction.

        Counter-gradient correction acts to sharpen species interfaces:

            J_cg = C_cg * D_eff * |grad(Y)| * grad(Y) / |grad(Y)|

        Parameters
        ----------
        grad_Y : torch.Tensor
            ``(n_cells, N, 3)`` mass fraction gradient.
        D_eff : torch.Tensor
            ``(n_cells, N)`` effective diffusivity.

        Returns
        -------
        torch.Tensor
            ``(n_cells, N, 3)`` counter-gradient correction flux.
        """
        C_cg = 0.1  # Counter-gradient coefficient
        grad_mag = grad_Y.norm(dim=-1, keepdim=True).clamp(min=_EPS)  # (n_cells, N, 1)
        D_expanded = D_eff.unsqueeze(-1)  # (n_cells, N, 1)
        J_cg = C_cg * D_expanded * grad_Y
        return J_cg

    def __repr__(self) -> str:
        sp = ", ".join(self._species)
        return (
            f"MulticomponentMixtureEnhanced2("
            f"n_species={self._n_species}, species=[{sp}], "
            f"has_Cp_poly={any(len(c) > 1 for c in self._Cp_poly)})"
        )
