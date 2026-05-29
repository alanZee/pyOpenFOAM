"""
Enhanced Wilke mixing rule with multi-component diffusion support.

Extends :class:`~pyfoam.thermophysical.wilke_transport.WilkeTransport` with:

- Multi-component binary diffusion coefficients
- Effective Lewis number computation
- Schmidt number per species pair
- Mass-fraction-based interface (in addition to mole-fraction)

Usage::

    from pyfoam.thermophysical.wilke_transport_enhanced import WilkeTransportEnhanced
    from pyfoam.thermophysical.transport_model import Sutherland

    wilke = WilkeTransportEnhanced(
        transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5, T_ref=273.15, S=139.0)],
        Mw=[28.014, 31.998],
        D_ij=[[0.0, 2.1e-5], [2.1e-5, 0.0]],
    )
    mu_mix = wilke.mu(T=300.0, x=[0.79, 0.21])
    D_eff = wilke.effective_diffusivity(T=300.0, x=[0.79, 0.21], species=0)
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.transport_model import TransportModel
from pyfoam.thermophysical.wilke_transport import WilkeTransport

__all__ = ["WilkeTransportEnhanced"]

logger = logging.getLogger(__name__)


class WilkeTransportEnhanced(WilkeTransport):
    """Enhanced Wilke mixing with binary diffusion coefficients.

    Extends :class:`WilkeTransport` with:

    - **Binary diffusion**: user-specified D_ij matrix for species pairs.
    - **Effective diffusivity**: :meth:`effective_diffusivity` computes
      the effective diffusion coefficient of a species in the mixture.
    - **Schmidt number**: :meth:`schmidt_number` per species.
    - **Lewis number**: :meth:`lewis_number` per species.
    - **Mass fraction interface**: :meth:`mu_from_mass_fractions` accepts
      mass fractions Y instead of mole fractions x.

    Parameters
    ----------
    transport_models : sequence of TransportModel
        One transport model per species.
    Mw : sequence of float
        Molecular weights (g/mol) for each species.
    D_ij : sequence of sequence of float or None
        Binary diffusion coefficients (m^2/s) at reference conditions.
        ``D_ij[i][j]`` is the diffusion coefficient of species i in j.
        If None, diffusion features are disabled.
    D_ref_T : float
        Reference temperature for D_ij (K). Default 298.15.
    D_ref_P : float
        Reference pressure for D_ij (Pa). Default 101325.

    Examples::

        from pyfoam.thermophysical.transport_model import Sutherland

        wilke = WilkeTransportEnhanced(
            transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
            Mw=[28.014, 31.998],
            D_ij=[[0.0, 2.1e-5], [2.1e-5, 0.0]],
        )
    """

    def __init__(
        self,
        transport_models: Sequence[TransportModel],
        Mw: Sequence[float],
        D_ij: Sequence[Sequence[float]] | None = None,
        D_ref_T: float = 298.15,
        D_ref_P: float = 101325.0,
    ) -> None:
        super().__init__(transport_models=transport_models, Mw=Mw)

        n = self._n_species

        if D_ij is not None:
            if len(D_ij) != n:
                raise ValueError(
                    f"D_ij rows ({len(D_ij)}) must match n_species ({n})"
                )
            for i, row in enumerate(D_ij):
                if len(row) != n:
                    raise ValueError(
                        f"D_ij[{i}] columns ({len(row)}) must match n_species ({n})"
                    )

        self._D_ij = (
            [list(row) for row in D_ij] if D_ij is not None else None
        )
        self._D_ref_T = D_ref_T
        self._D_ref_P = D_ref_P

    @property
    def has_diffusion(self) -> bool:
        """Whether binary diffusion data is available."""
        return self._D_ij is not None

    # ------------------------------------------------------------------
    # Temperature-dependent diffusion
    # ------------------------------------------------------------------

    def D_ij(self, i: int, j: int, T: float, P: float = 101325.0) -> float:
        """Binary diffusion coefficient of species i in j at (T, P).

        Uses the classic scaling: D_ij(T, P) = D_ij_ref * (T/T_ref)^1.75 * (P_ref/P)

        Parameters
        ----------
        i : int
            Species index.
        j : int
            Species index.
        T : float
            Temperature (K).
        P : float
            Pressure (Pa). Default 101325.

        Returns
        -------
        float
            Binary diffusion coefficient (m^2/s).

        Raises
        ------
        RuntimeError
            If no diffusion data is available.
        """
        if self._D_ij is None:
            raise RuntimeError("No diffusion data (D_ij) available")

        D_ref = self._D_ij[i][j]
        T_ratio = (T / self._D_ref_T) ** 1.75
        P_ratio = self._D_ref_P / max(P, 1.0)
        return D_ref * T_ratio * P_ratio

    # ------------------------------------------------------------------
    # Effective diffusivity
    # ------------------------------------------------------------------

    def effective_diffusivity(
        self,
        T: float,
        x: Sequence[float],
        species: int,
        P: float = 101325.0,
    ) -> float:
        """Effective diffusivity of a species in the mixture.

        Uses the Wilke approximation:

            D_eff_k = (1 - x_k) / sum_{j!=k} x_j / D_kj

        Parameters
        ----------
        T : float
            Temperature (K).
        x : sequence of float
            Mole fractions.
        species : int
            Index of the species to compute diffusivity for.
        P : float
            Pressure (Pa). Default 101325.

        Returns
        -------
        float
            Effective diffusivity (m^2/s).
        """
        if self._D_ij is None:
            raise RuntimeError("No diffusion data available")
        if len(x) != self._n_species:
            raise ValueError("x length must match n_species")

        x_k = x[species]
        numerator = 1.0 - x_k
        denominator = 0.0
        for j in range(self._n_species):
            if j == species:
                continue
            D_kj = self.D_ij(species, j, T, P)
            if D_kj > 0:
                denominator += x[j] / D_kj

        if denominator < 1e-30:
            return 0.0

        return numerator / denominator

    # ------------------------------------------------------------------
    # Schmidt and Lewis numbers
    # ------------------------------------------------------------------

    def schmidt_number(
        self,
        T: float,
        x: Sequence[float],
        species: int,
        rho: float = 1.2,
        P: float = 101325.0,
    ) -> float:
        """Schmidt number of a species in the mixture.

        Sc = mu_mix / (rho * D_eff)

        Parameters
        ----------
        T : float
            Temperature (K).
        x : sequence of float
            Mole fractions.
        species : int
            Species index.
        rho : float
            Mixture density (kg/m^3).
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Schmidt number (dimensionless).
        """
        mu = float(self.mu(T, x).item())
        D = self.effective_diffusivity(T, x, species, P)
        if D < 1e-30:
            return float("inf")
        return mu / (rho * D)

    def lewis_number(
        self,
        T: float,
        x: Sequence[float],
        species: int,
        Cp: float = 1005.0,
        kappa: float | None = None,
        rho: float = 1.2,
        P: float = 101325.0,
    ) -> float:
        """Lewis number of a species in the mixture.

        Le = kappa / (rho * Cp * D_eff)

        Parameters
        ----------
        T : float
            Temperature (K).
        x : sequence of float
            Mole fractions.
        species : int
            Species index.
        Cp : float
            Specific heat (J/(kg·K)).
        kappa : float or None
            Thermal conductivity (W/(m·K)). If None, computed from
            Wilke mixing rule.
        rho : float
            Density (kg/m^3).
        P : float
            Pressure (Pa).

        Returns
        -------
        float
            Lewis number (dimensionless).
        """
        if kappa is None:
            kappa = float(self.kappa(T, x, Cp=Cp).item())
        D = self.effective_diffusivity(T, x, species, P)
        if D < 1e-30:
            return float("inf")
        return kappa / (rho * Cp * D)

    # ------------------------------------------------------------------
    # Mass fraction interface
    # ------------------------------------------------------------------

    def mu_from_mass_fractions(
        self,
        T: torch.Tensor | float,
        Y: Sequence[float],
    ) -> torch.Tensor:
        """Compute mixture viscosity from mass fractions.

        Converts mass fractions to mole fractions and delegates to
        :meth:`WilkeTransport.mu`.

        Parameters
        ----------
        T : float or torch.Tensor
            Temperature (K).
        Y : sequence of float
            Mass fractions for each species.

        Returns
        -------
        torch.Tensor
            Mixture dynamic viscosity (Pa·s).
        """
        if len(Y) != self._n_species:
            raise ValueError("Y length must match n_species")

        # Convert mass fractions to mole fractions
        # x_i = (Y_i / M_i) / sum(Y_j / M_j)
        y_over_m = [Y[i] / self._Mw[i] for i in range(self._n_species)]
        total = sum(y_over_m)
        if total < 1e-30:
            total = 1e-30
        x = [yom / total for yom in y_over_m]

        return self.mu(T, x)

    def __repr__(self) -> str:
        model_names = [type(m).__name__ for m in self._models]
        has_diff = "with diffusion" if self.has_diffusion else "no diffusion"
        return (
            f"WilkeTransportEnhanced(n_species={self._n_species}, "
            f"models={model_names}, {has_diff})"
        )
