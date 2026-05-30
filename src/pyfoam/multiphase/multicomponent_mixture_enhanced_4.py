"""
Enhanced multicomponent mixture property model — v5.

在 Enhanced v4 基础上增加：

- **Soret/Dufour 耦合**：热扩散和扩散热效应
- **高阶 Cp 多项式**：支持 NASA 七系数多项式
- **质量加权混合规则**：改进的 Wilke 混合粘度

Usage::

    from pyfoam.multiphase.multicomponent_mixture_enhanced_4 import (
        MulticomponentMixtureEnhanced4,
    )

    mix = MulticomponentMixtureEnhanced4(
        species=["N2", "O2", "H2O"],
        M=[28.014e-3, 32.0e-3, 18.015e-3],
        rho=[1.165, 1.331, 0.804],
        mu=[1.76e-5, 2.04e-5, 0.96e-5],
        Cp=[1040.0, 919.0, 2080.0],
        D=[2.1e-5, 2.1e-5, 2.5e-5],
        Sc_t=[0.7, 0.7, 0.7],
        Le=[1.0, 1.0, 0.8],
        soret_coeff=[0.01, 0.01, 0.05],
    )
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.multicomponent_mixture_enhanced_3 import (
    MulticomponentMixtureEnhanced3,
)

__all__ = ["MulticomponentMixtureEnhanced4"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class MulticomponentMixtureEnhanced4(MulticomponentMixtureEnhanced3):
    """Enhanced multicomponent mixture v5 with Soret/Dufour and improved mixing rules.

    在 v4 基础上增加：
    - Soret/Dufour 耦合效应
    - NASA 七系数 Cp 多项式支持
    - Wilke 混合粘度规则

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
    ) -> None:
        super().__init__(
            species, M, rho, mu, Cp, kappa, D, Sc_t, Cp_poly,
            Le, reaction_rates,
        )

        # Soret coefficients
        if soret_coeff is not None:
            if len(soret_coeff) != self._n_species:
                raise ValueError(
                    f"soret_coeff length ({len(soret_coeff)}) != n_species ({self._n_species})"
                )
            self._soret_coeff = list(soret_coeff)
        else:
            self._soret_coeff = [0.0] * self._n_species

        # Dufour coefficients
        if dufour_coeff is not None:
            if len(dufour_coeff) != self._n_species:
                raise ValueError(
                    f"dufour_coeff length ({len(dufour_coeff)}) != n_species ({self._n_species})"
                )
            self._dufour_coeff = list(dufour_coeff)
        else:
            self._dufour_coeff = [0.0] * self._n_species

    @property
    def soret_coeff(self) -> list[float]:
        """Soret (thermal diffusion) coefficients."""
        return self._soret_coeff.copy()

    @property
    def dufour_coeff(self) -> list[float]:
        """Dufour (diffusion thermo) coefficients."""
        return self._dufour_coeff.copy()

    # ------------------------------------------------------------------
    # Soret 热扩散
    # ------------------------------------------------------------------

    def soret_flux(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        grad_T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Soret (thermal diffusion) mass flux.

        J_Soret_i = -D_T_i * Y_i * grad(T) / T

        where D_T_i = soret_coeff_i * D_mol_i.

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).
        grad_T : torch.Tensor
            ``(n_cells, 3)`` temperature gradient (K/m).

        Returns
        -------
        torch.Tensor
            ``(n_cells, N, 3)`` Soret mass flux (kg/(m^2·s)).
        """
        device = Y.device
        dtype = Y.dtype
        n_cells = Y.shape[0]
        n_sp = self._n_species

        soret = torch.tensor(self._soret_coeff, device=device, dtype=dtype)
        D_mol = torch.tensor(self._D, device=device, dtype=dtype)
        D_T = soret * D_mol  # (N,)

        T_safe = T.to(device=device, dtype=dtype).clamp(min=_EPS)
        grad_T_dev = grad_T.to(device=device, dtype=dtype)  # (n_cells, 3)

        # J_Soret_i = -D_T_i * Y_i * grad(T) / T
        # Shape: (n_cells, N, 3)
        flux = torch.zeros(n_cells, n_sp, 3, device=device, dtype=dtype)
        for i in range(n_sp):
            flux[:, i, :] = -D_T[i] * Y[:, i].unsqueeze(-1) * grad_T_dev / T_safe.unsqueeze(-1)

        return flux

    # ------------------------------------------------------------------
    # Dufour 扩散热
    # ------------------------------------------------------------------

    def dufour_heat_flux(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        grad_Y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dufour (diffusion thermo) heat flux.

        q_Dufour = sum_i Duf_i * D_mol_i * grad(Y_i) * T

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).
        grad_Y : torch.Tensor
            ``(n_cells, N, 3)`` species mass fraction gradients.

        Returns
        -------
        torch.Tensor
            ``(n_cells, 3)`` Dufour heat flux (W/m^2).
        """
        device = Y.device
        dtype = Y.dtype
        n_cells = Y.shape[0]

        dufour = torch.tensor(self._dufour_coeff, device=device, dtype=dtype)
        D_mol = torch.tensor(self._D, device=device, dtype=dtype)
        T_safe = T.to(device=device, dtype=dtype).clamp(min=_EPS)

        grad_Y_dev = grad_Y.to(device=device, dtype=dtype)  # (n_cells, N, 3)

        q = torch.zeros(n_cells, 3, device=device, dtype=dtype)
        for i in range(self._n_species):
            q = q + dufour[i] * D_mol[i] * grad_Y_dev[:, i, :] * T_safe.unsqueeze(-1)

        return q

    # ------------------------------------------------------------------
    # Wilke 混合粘度
    # ------------------------------------------------------------------

    def wilke_viscosity(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture viscosity using Wilke's semi-empirical rule.

        mu_m = sum_i (X_i * mu_i) / sum_j (X_j * phi_ij)

        phi_ij = (1 + (mu_i/mu_j)^(1/2) * (M_j/M_i)^(1/4))^2
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

        X = self.mass_to_mole(Y)
        mu_pure = torch.tensor(self._mu, device=device, dtype=dtype)
        M_arr = torch.tensor(self._M, device=device, dtype=dtype)

        mu_mix = torch.zeros(n_cells, device=device, dtype=dtype)

        for i in range(n_sp):
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

    def __repr__(self) -> str:
        sp = ", ".join(self._species)
        has_soret = any(s != 0 for s in self._soret_coeff)
        has_dufour = any(d != 0 for d in self._dufour_coeff)
        return (
            f"MulticomponentMixtureEnhanced4("
            f"n_species={self._n_species}, species=[{sp}], "
            f"has_soret={has_soret}, has_dufour={has_dufour})"
        )
