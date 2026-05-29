"""
Enhanced Sutherland transport model with multi-species support.

Extends :class:`~pyfoam.thermophysical.sutherland_transport.SutherlandTransport`
with:

- Multi-species Sutherland coefficients (per-species mu_ref, T_ref, S)
- Mixture viscosity via Wilke-style mixing from species viscosities
- Species-specific thermal conductivity

Usage::

    from pyfoam.thermophysical.sutherland_transport_enhanced import SutherlandTransportEnhanced

    transport = SutherlandTransportEnhanced(
        species_params=[
            {"name": "N2", "mu_ref": 1.663e-5, "T_ref": 273.15, "S": 107.0, "Mw": 28.014},
            {"name": "O2", "mu_ref": 1.919e-5, "T_ref": 273.15, "S": 139.0, "Mw": 31.998},
        ],
    )
    mu_N2 = transport.species_mu("N2", T=300.0)
    mu_mix = transport.mu(T=300.0, x=[0.79, 0.21])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.sutherland_transport import SutherlandTransport

__all__ = ["SutherlandTransportEnhanced", "SpeciesSutherlandParams"]

logger = logging.getLogger(__name__)


@dataclass
class SpeciesSutherlandParams:
    """Sutherland parameters for a single species.

    Parameters
    ----------
    name : str
        Species name (e.g. "N2", "O2").
    mu_ref : float
        Reference viscosity (Pa·s).
    T_ref : float
        Reference temperature (K).
    S : float
        Sutherland constant (K).
    Mw : float
        Molecular weight (g/mol).
    kappa_coeffs : list of float or None
        Polynomial coefficients for thermal conductivity (optional).
    """

    name: str
    mu_ref: float = 1.716e-5
    T_ref: float = 273.15
    S: float = 110.4
    Mw: float = 28.964
    kappa_coeffs: list[float] | None = None

    def __post_init__(self) -> None:
        if self.mu_ref <= 0:
            raise ValueError(f"mu_ref must be positive, got {self.mu_ref}")
        if self.T_ref <= 0:
            raise ValueError(f"T_ref must be positive, got {self.T_ref}")
        if self.S <= 0:
            raise ValueError(f"S must be positive, got {self.S}")
        if self.Mw <= 0:
            raise ValueError(f"Mw must be positive, got {self.Mw}")


class SutherlandTransportEnhanced(SutherlandTransport):
    """Multi-species Sutherland transport model.

    Provides per-species viscosity computation using individual
    Sutherland parameters, and mixture viscosity via the Wilke
    semi-empirical mixing rule.

    Can also operate in single-species mode (inherited from
    :class:`SutherlandTransport`).

    Parameters
    ----------
    species_params : sequence of SpeciesSutherlandParams, optional
        Per-species Sutherland parameters. If provided, enables
        multi-species mode.
    mu_ref : float
        Single-species reference viscosity (ignored if species_params given).
    T_ref : float
        Single-species reference temperature.
    S : float
        Single-species Sutherland constant.
    kappa_coeffs : sequence of float or None
        Single-species kappa polynomial.
    """

    def __init__(
        self,
        species_params: Sequence[SpeciesSutherlandParams] | None = None,
        mu_ref: float = 1.716e-5,
        T_ref: float = 273.15,
        S: float = 110.4,
        kappa_coeffs: Sequence[float] | None = None,
    ) -> None:
        super().__init__(
            mu_ref=mu_ref, T_ref=T_ref, S=S, kappa_coeffs=kappa_coeffs
        )

        self._species_params = list(species_params) if species_params else None
        self._species_names: list[str] = []

        if self._species_params:
            self._species_names = [sp.name for sp in self._species_params]
            self._n_species = len(self._species_params)
        else:
            self._n_species = 0

    @property
    def n_species(self) -> int:
        """Number of species in multi-species mode."""
        return self._n_species

    @property
    def species_names(self) -> list[str]:
        """Species names."""
        return self._species_names.copy()

    @property
    def is_multispecies(self) -> bool:
        """Whether multi-species mode is active."""
        return self._n_species > 0

    # ------------------------------------------------------------------
    # Per-species viscosity
    # ------------------------------------------------------------------

    def species_mu(self, name: str, T: torch.Tensor | float) -> torch.Tensor:
        """Compute pure-component viscosity for a named species.

        Parameters
        ----------
        name : str
            Species name.
        T : float or torch.Tensor
            Temperature (K).

        Returns
        -------
        torch.Tensor
            Dynamic viscosity (Pa·s).

        Raises
        ------
        KeyError
            If species name not found.
        """
        if not self.is_multispecies:
            raise RuntimeError("Not in multi-species mode")

        idx = self._get_species_index(name)
        sp = self._species_params[idx]

        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        T_safe = T.clamp(min=1.0)
        T_ratio = T_safe / sp.T_ref
        return (
            sp.mu_ref
            * T_ratio.pow(1.5)
            * (sp.T_ref + sp.S)
            / (T_safe + sp.S)
        )

    def _get_species_index(self, name: str) -> int:
        """Get species index by name."""
        for i, n in enumerate(self._species_names):
            if n == name:
                return i
        raise KeyError(
            f"Species '{name}' not found. Available: {self._species_names}"
        )

    # ------------------------------------------------------------------
    # Wilke mixture viscosity
    # ------------------------------------------------------------------

    def mu(
        self,
        T: torch.Tensor | float,
        x: Sequence[float] | None = None,
    ) -> torch.Tensor:
        """Compute viscosity.

        In single-species mode, returns pure-component viscosity.
        In multi-species mode, uses Wilke mixing rule with mole fractions.

        Parameters
        ----------
        T : float or torch.Tensor
            Temperature (K).
        x : sequence of float or None
            Mole fractions (required in multi-species mode).

        Returns
        -------
        torch.Tensor
            Dynamic viscosity (Pa·s).
        """
        if not self.is_multispecies:
            return super().mu(T)

        if x is None:
            raise ValueError("Mole fractions x required in multi-species mode")
        if len(x) != self._n_species:
            raise ValueError("x length must match n_species")

        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=dtype, device=device)

        # Compute pure-component viscosities
        mu_pure = []
        for sp in self._species_params:
            T_safe = T.clamp(min=1.0)
            T_ratio = T_safe / sp.T_ref
            mu_i = sp.mu_ref * T_ratio.pow(1.5) * (sp.T_ref + sp.S) / (T_safe + sp.S)
            mu_pure.append(mu_i)

        # Wilke mixing
        Mw = [sp.Mw for sp in self._species_params]
        n = self._n_species
        x_t = torch.tensor(x, dtype=dtype, device=device)

        # Build interaction matrix Phi_ij
        is_batch = mu_pure[0].dim() > 0
        if is_batch:
            n_cells = mu_pure[0].shape[0]
            Phi = torch.zeros(n, n, n_cells, dtype=dtype, device=device)
        else:
            Phi = torch.zeros(n, n, dtype=dtype, device=device)

        for i in range(n):
            for j in range(n):
                Mw_ratio = Mw[i] / Mw[j]
                mu_ratio = mu_pure[i] / mu_pure[j].clamp(min=1e-30)
                Phi[i, j] = (
                    (1.0 / 8.0**0.5)
                    * (1.0 + Mw_ratio) ** (-0.5)
                    * (1.0 + mu_ratio.sqrt() * (Mw[j] / Mw[i]) ** 0.25) ** 2
                )

        # mu_mix = sum_i (x_i * mu_i) / sum_j (x_j * Phi_ij)
        if is_batch:
            denom = torch.einsum("ijc,j->ic", Phi, x_t)
            mu_stack = torch.stack(mu_pure, dim=0)
            numerator = (x_t.unsqueeze(-1) * mu_stack).sum(dim=0)
            denominator = (x_t.unsqueeze(-1) * denom).sum(dim=0)
        else:
            denom = torch.matmul(Phi, x_t)
            mu_stack = torch.stack(mu_pure, dim=0)
            numerator = (x_t * mu_stack).sum()
            denominator = (x_t * denom).sum()

        return numerator / denominator.clamp(min=1e-30)

    def __repr__(self) -> str:
        if self.is_multispecies:
            return (
                f"SutherlandTransportEnhanced(n_species={self._n_species}, "
                f"species={self._species_names})"
            )
        return (
            f"SutherlandTransportEnhanced(mu_ref={self._mu_ref}, "
            f"T_ref={self._T_ref}, S={self._S}, single-species)"
        )
