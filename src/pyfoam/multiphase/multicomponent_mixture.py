"""
Multi-component mixture property model.

Computes thermophysical properties for a mixture of N species, each with
its own density, viscosity, specific heat, and molecular weight.  Properties
are computed as mass-fraction-weighted (or mole-fraction-weighted) sums.

Mixture rules:

    rho_m   = 1 / sum(Y_i / rho_i)              (volume-weighted)
    mu_m    = sum(Y_i * mu_i)                     (mass-weighted)
    Cp_m    = sum(Y_i * Cp_i)                     (mass-weighted)
    M_m     = 1 / sum(Y_i / M_i)                  (molecular weight)
    Y_i     = X_i * M_i / M_m                     (mass from mole fraction)

Used in OpenFOAM's ``multiComponentMixture`` template class.

Usage::

    from pyfoam.multiphase.multicomponent_mixture import MulticomponentMixture

    mix = MulticomponentMixture(
        species=["N2", "O2", "H2O"],
        M=[28.014e-3, 32.0e-3, 18.015e-3],
        rho=[1.165, 1.331, 0.804],
        mu=[1.76e-5, 2.04e-5, 0.96e-5],
        Cp=[1040.0, 919.0, 2080.0],
    )
    rho_m = mix.mixture_density(Y)
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["MulticomponentMixture"]

logger = logging.getLogger(__name__)


class MulticomponentMixture:
    """Multi-component mixture property model.

    Computes mass-fraction-weighted mixture properties for N species.

    Parameters
    ----------
    species : sequence of str
        Species names (N >= 1).
    M : sequence of float
        Molecular weight of each species (kg/mol).
    rho : sequence of float
        Pure-species density at reference conditions (kg/m^3).
    mu : sequence of float
        Pure-species dynamic viscosity (Pa·s).
    Cp : sequence of float
        Pure-species specific heat at constant pressure (J/(kg·K)).
    kappa : sequence of float, optional
        Pure-species thermal conductivity (W/(m·K)).
        If None, computed from mu, Cp, and Pr = 0.7.

    Examples::

        mix = MulticomponentMixture(
            ["N2", "O2"],
            M=[28.014e-3, 32.0e-3],
            rho=[1.165, 1.331],
            mu=[1.76e-5, 2.04e-5],
            Cp=[1040.0, 919.0],
        )
    """

    def __init__(
        self,
        species: Sequence[str],
        M: Sequence[float],
        rho: Sequence[float],
        mu: Sequence[float],
        Cp: Sequence[float],
        kappa: Sequence[float] | None = None,
    ) -> None:
        n_species = len(species)
        if n_species < 1:
            raise ValueError("Need at least 1 species")
        for arr, name in [(M, "M"), (rho, "rho"), (mu, "mu"), (Cp, "Cp")]:
            if len(arr) != n_species:
                raise ValueError(f"{name} length ({len(arr)}) != n_species ({n_species})")

        self._n_species = n_species
        self._species = list(species)
        self._M = list(M)
        self._rho = list(rho)
        self._mu = list(mu)
        self._Cp = list(Cp)
        self._kappa = list(kappa) if kappa is not None else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_species(self) -> int:
        return self._n_species

    @property
    def species(self) -> list[str]:
        return self._species.copy()

    @property
    def M(self) -> list[float]:
        """Molecular weights (kg/mol)."""
        return self._M.copy()

    # ------------------------------------------------------------------
    # Fraction conversions
    # ------------------------------------------------------------------

    def validate_mass_fractions(self, Y: torch.Tensor) -> torch.Tensor:
        """Clamp and renormalise mass fractions so sum(Y) = 1.

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.

        Returns
        -------
        torch.Tensor
            Normalised mass fractions.
        """
        Y = Y.clamp(min=0.0)
        total = Y.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        return Y / total

    def mole_to_mass(self, X: torch.Tensor) -> torch.Tensor:
        """Convert mole fractions to mass fractions.

        Y_i = X_i * M_i / M_m  where M_m = 1 / sum(X_i / M_i)

        Parameters
        ----------
        X : torch.Tensor
            ``(n_cells, N)`` mole fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells, N)`` mass fractions.
        """
        X = X.clamp(min=0.0)
        X_total = X.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        X = X / X_total

        device = X.device
        dtype = X.dtype
        M = torch.tensor(self._M, device=device, dtype=dtype)

        # Mean molecular weight: M_m = sum(X_i * M_i)
        M_m = (X * M).sum(dim=-1, keepdim=True).clamp(min=1e-30)

        Y = X * M / M_m
        Y_total = Y.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        return Y / Y_total

    def mass_to_mole(self, Y: torch.Tensor) -> torch.Tensor:
        """Convert mass fractions to mole fractions.

        X_i = (Y_i / M_i) / sum(Y_j / M_j)

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells, N)`` mole fractions.
        """
        Y = self.validate_mass_fractions(Y)

        device = Y.device
        dtype = Y.dtype
        M = torch.tensor(self._M, device=device, dtype=dtype)

        n_i = Y / M.clamp(min=1e-30)
        n_total = n_i.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        return n_i / n_total

    # ------------------------------------------------------------------
    # Mixture properties
    # ------------------------------------------------------------------

    def mixture_density(self, Y: torch.Tensor) -> torch.Tensor:
        """Mixture density via volume-averaged rule.

        rho_m = 1 / sum(Y_i / rho_i)

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture density (kg/m^3).
        """
        Y = self.validate_mass_fractions(Y)
        device = Y.device
        dtype = Y.dtype
        rho = torch.tensor(self._rho, device=device, dtype=dtype)

        inv_rho = Y / rho.clamp(min=1e-30)
        return 1.0 / inv_rho.sum(dim=-1).clamp(min=1e-30)

    def mixture_viscosity(self, Y: torch.Tensor) -> torch.Tensor:
        """Mixture viscosity (mass-weighted).

        mu_m = sum(Y_i * mu_i)

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture viscosity (Pa·s).
        """
        Y = self.validate_mass_fractions(Y)
        device = Y.device
        dtype = Y.dtype
        mu = torch.tensor(self._mu, device=device, dtype=dtype)
        return (Y * mu).sum(dim=-1)

    def mixture_Cp(self, Y: torch.Tensor) -> torch.Tensor:
        """Mixture specific heat (mass-weighted).

        Cp_m = sum(Y_i * Cp_i)

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture Cp (J/(kg·K)).
        """
        Y = self.validate_mass_fractions(Y)
        device = Y.device
        dtype = Y.dtype
        Cp = torch.tensor(self._Cp, device=device, dtype=dtype)
        return (Y * Cp).sum(dim=-1)

    def mixture_kappa(self, Y: torch.Tensor, Pr: float = 0.7) -> torch.Tensor:
        """Mixture thermal conductivity.

        If pure-species kappa were provided, uses mass-weighted sum.
        Otherwise: kappa_m = mu_m * Cp_m / Pr.

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        Pr : float
            Prandtl number (used if kappa not provided).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture thermal conductivity (W/(m·K)).
        """
        if self._kappa is not None:
            Y = self.validate_mass_fractions(Y)
            device = Y.device
            dtype = Y.dtype
            kappa = torch.tensor(self._kappa, device=device, dtype=dtype)
            return (Y * kappa).sum(dim=-1)

        return self.mixture_viscosity(Y) * self.mixture_Cp(Y) / Pr

    def mixture_M(self, Y: torch.Tensor) -> torch.Tensor:
        """Mixture molecular weight from mass fractions.

        M_m = 1 / sum(Y_i / M_i)

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture molecular weight (kg/mol).
        """
        Y = self.validate_mass_fractions(Y)
        device = Y.device
        dtype = Y.dtype
        M = torch.tensor(self._M, device=device, dtype=dtype)
        inv_M = Y / M.clamp(min=1e-30)
        return 1.0 / inv_M.sum(dim=-1).clamp(min=1e-30)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def compute_all(
        self, Y: torch.Tensor, Pr: float = 0.7,
    ) -> dict[str, torch.Tensor]:
        """Compute all mixture properties in one call.

        Parameters
        ----------
        Y : torch.Tensor
            ``(n_cells, N)`` mass fractions.
        Pr : float
            Prandtl number.

        Returns
        -------
        dict
            Keys: ``rho_m``, ``mu_m``, ``Cp_m``, ``kappa_m``, ``M_m``.
        """
        return {
            "rho_m": self.mixture_density(Y),
            "mu_m": self.mixture_viscosity(Y),
            "Cp_m": self.mixture_Cp(Y),
            "kappa_m": self.mixture_kappa(Y, Pr),
            "M_m": self.mixture_M(Y),
        }

    def __repr__(self) -> str:
        sp = ", ".join(self._species)
        return f"MulticomponentMixture(n_species={self._n_species}, species=[{sp}])"
