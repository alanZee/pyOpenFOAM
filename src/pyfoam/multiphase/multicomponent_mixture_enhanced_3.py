"""
Enhanced multicomponent mixture property model — v4.

在 Enhanced v3 基础上增加：

- **活性组分输运**：支持带反应源项的组分输运
- **增强混合热导率**：基于 Wassiljewa 方程的多组分热导率
- **修正 Lewis 数**：改进的非单位 Lewis 数处理

Usage::

    from pyfoam.multiphase.multicomponent_mixture_enhanced_3 import (
        MulticomponentMixtureEnhanced3,
    )

    mix = MulticomponentMixtureEnhanced3(
        species=["N2", "O2", "H2O"],
        M=[28.014e-3, 32.0e-3, 18.015e-3],
        rho=[1.165, 1.331, 0.804],
        mu=[1.76e-5, 2.04e-5, 0.96e-5],
        Cp=[1040.0, 919.0, 2080.0],
        D=[2.1e-5, 2.1e-5, 2.5e-5],
        Sc_t=[0.7, 0.7, 0.7],
        Le=[1.0, 1.0, 0.8],
    )
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.multiphase.multicomponent_mixture_enhanced_2 import (
    MulticomponentMixtureEnhanced2,
)

__all__ = ["MulticomponentMixtureEnhanced3"]

logger = logging.getLogger(__name__)

_EPS = 1e-30


class MulticomponentMixtureEnhanced3(MulticomponentMixtureEnhanced2):
    """Enhanced multicomponent mixture v4 with reactive transport and improved thermodynamics.

    在 v3 基础上增加：
    - 活性组分输运（带反应源项）
    - Wassiljewa 混合热导率
    - 非单位 Lewis 数处理

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
        Lewis numbers per species. Default ``[1.0, ...]``.
    reaction_rates : sequence of float, optional
        Volumetric reaction rate constants per species (1/s).
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
    ) -> None:
        super().__init__(species, M, rho, mu, Cp, kappa, D, Sc_t, Cp_poly)

        # Lewis numbers
        if Le is not None:
            if len(Le) != self._n_species:
                raise ValueError(
                    f"Le length ({len(Le)}) != n_species ({self._n_species})"
                )
            self._Le = [max(l, _EPS) for l in Le]
        else:
            self._Le = [1.0] * self._n_species

        # Reaction rates
        if reaction_rates is not None:
            if len(reaction_rates) != self._n_species:
                raise ValueError(
                    f"reaction_rates length ({len(reaction_rates)}) "
                    f"!= n_species ({self._n_species})"
                )
            self._reaction_rates = list(reaction_rates)
        else:
            self._reaction_rates = [0.0] * self._n_species

    @property
    def Le(self) -> list[float]:
        """Lewis numbers per species."""
        return self._Le.copy()

    @property
    def reaction_rates(self) -> list[float]:
        """Reaction rate constants per species."""
        return self._reaction_rates.copy()

    # ------------------------------------------------------------------
    # 活性组分输运
    # ------------------------------------------------------------------

    def reaction_source(
        self,
        Y: torch.Tensor,
        rho_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reaction source terms for species transport.

        S_i = k_i * rho_m * Y_i  (first-order reaction)

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        rho_mix : torch.Tensor
            ``(n_cells,)`` mixture density (kg/m^3).

        Returns
        -------
        torch.Tensor
            ``(n_cells, N)`` reaction source terms (kg/(m^3·s)).
        """
        device = Y.device
        dtype = Y.dtype
        n_cells = Y.shape[0]

        k = torch.tensor(self._reaction_rates, device=device, dtype=dtype)  # (N,)
        rho_expanded = rho_mix.unsqueeze(-1)  # (n_cells, 1)

        # First-order: S = k * rho * Y
        S = k.unsqueeze(0) * rho_expanded * Y

        return S

    # ------------------------------------------------------------------
    # Wassiljewa 混合热导率
    # ------------------------------------------------------------------

    def wassiljewa_conductivity(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture thermal conductivity using Wassiljewa equation.

        kappa_m = sum_i (X_i * kappa_i) / sum_j (X_j * A_ij)

        where A_ij = (1 + (kappa_i/kappa_j)^(1/2) * (M_j/M_i)^(1/4))^2
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
            ``(n_cells,)`` mixture thermal conductivity (W/(m·K)).
        """
        if self._kappa is None:
            # Fallback: use a default value
            return torch.full((Y.shape[0],), 0.026, device=Y.device, dtype=Y.dtype)

        Y = self.validate_mass_fractions(Y)
        device = Y.device
        dtype = Y.dtype
        n_cells = Y.shape[0]
        n_sp = self._n_species

        # Mole fractions
        X = self.mass_to_mole(Y)

        # Pure-species conductivities
        kappa_pure = torch.tensor(self._kappa, device=device, dtype=dtype)
        M_arr = torch.tensor(self._M, device=device, dtype=dtype)

        kappa_mix = torch.zeros(n_cells, device=device, dtype=dtype)

        for i in range(n_sp):
            denom = torch.zeros(n_cells, device=device, dtype=dtype)
            for j in range(n_sp):
                if i == j:
                    denom = denom + X[:, j]
                else:
                    kappa_ratio = (kappa_pure[i] / kappa_pure[j].clamp(min=_EPS)).sqrt()
                    M_ratio = (M_arr[j] / M_arr[i].clamp(min=_EPS)).pow(0.25)
                    A_ij = (1.0 + kappa_ratio * M_ratio).pow(2)
                    A_ij = A_ij / (8.0 * (1.0 + M_arr[i] / M_arr[j].clamp(min=_EPS))).sqrt()
                    denom = denom + X[:, j] * A_ij

            kappa_mix = kappa_mix + X[:, i] * kappa_pure[i] / denom.clamp(min=_EPS)

        return kappa_mix.clamp(min=_EPS)

    # ------------------------------------------------------------------
    # 修正 Lewis 数扩散
    # ------------------------------------------------------------------

    def effective_diffusivity(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        D_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute effective species diffusivity with Lewis number correction.

        D_eff_i = D_mol_i / Le_i + D_turb / Sc_t_i

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        T : torch.Tensor
            ``(n_cells,)`` temperature (K).
        D_t : torch.Tensor, optional
            ``(n_cells,)`` turbulent diffusivity. If None, only molecular.

        Returns
        -------
        torch.Tensor
            ``(n_cells, N)`` effective diffusivity per species (m^2/s).
        """
        device = Y.device
        dtype = Y.dtype
        n_cells = Y.shape[0]
        n_sp = self._n_species

        Le_t = torch.tensor(self._Le, device=device, dtype=dtype)  # (N,)
        D_mol = torch.tensor(self._D, device=device, dtype=dtype)  # (N,)

        # Molecular contribution with Le correction
        D_eff = D_mol.unsqueeze(0) / Le_t.unsqueeze(0)  # (1, N)
        D_eff = D_eff.expand(n_cells, -1).clone()

        # Turbulent contribution
        if D_t is not None:
            Sc_t = torch.tensor(self._Sc_t, device=device, dtype=dtype)
            D_turb = D_t.to(device=device, dtype=dtype).unsqueeze(-1)  # (n_cells, 1)
            D_eff = D_eff + D_turb / Sc_t.unsqueeze(0)

        return D_eff.clamp(min=_EPS)

    def __repr__(self) -> str:
        sp = ", ".join(self._species)
        has_reaction = any(k != 0 for k in self._reaction_rates)
        return (
            f"MulticomponentMixtureEnhanced3("
            f"n_species={self._n_species}, species=[{sp}], "
            f"has_Cp_poly={any(len(c) > 1 for c in self._Cp_poly)}, "
            f"has_reaction={has_reaction})"
        )
