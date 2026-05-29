"""
N-phase Volume of Fluid (VOF) advection for incompressible multiphase flows.

Extends the two-phase VOF method to N incompressible phases with a
single constraint: sum(alpha_i) = 1.  Each phase volume fraction is
transported with interface compression and bounded to [0, 1].

Governing equations for each phase i (1..N-1):

    d(alpha_i)/dt + div(U * alpha_i) + div(U_r * alpha_i * (1 - alpha_i)) = 0

The N-th phase volume fraction is computed from the constraint:

    alpha_N = 1 - sum(alpha_1 .. alpha_{N-1})

Mixture properties:

    rho_m  = sum(alpha_i * rho_i)
    mu_m   = sum(alpha_i * mu_i)
    nu_m   = mu_m / rho_m

Usage::

    from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

    model = IncompressibleMultiphaseVoF(
        phase_names=["water", "air", "oil"],
        rho=[998.0, 1.225, 850.0],
        mu=[1.002e-3, 1.8e-5, 0.03],
    )
    rho_m = model.mixture_density(alphas)
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["IncompressibleMultiphaseVoF"]

logger = logging.getLogger(__name__)


class IncompressibleMultiphaseVoF:
    """N-phase VOF model for incompressible multiphase flows.

    Manages N volume fractions (N-1 independent, the N-th derived from
    the summation constraint), advects each independently with interface
    compression, and computes mixture properties.

    Parameters
    ----------
    phase_names : sequence of str
        Names of the N phases (N >= 2).
    rho : sequence of float
        Density of each phase (kg/m^3).
    mu : sequence of float
        Dynamic viscosity of each phase (Pa·s).
    C_alpha : float
        Compression coefficient (0 = none, 1 = full). Default 1.0.

    Examples::

        model = IncompressibleMultiphaseVoF(
            ["water", "air"], [998.0, 1.225], [1.002e-3, 1.8e-5],
        )
        alphas = model.advance(alphas, phi, U, delta_t, mesh)
    """

    def __init__(
        self,
        phase_names: Sequence[str],
        rho: Sequence[float],
        mu: Sequence[float],
        C_alpha: float = 1.0,
    ) -> None:
        n_phases = len(phase_names)
        if n_phases < 2:
            raise ValueError("Need at least 2 phases")
        if len(rho) != n_phases:
            raise ValueError(f"rho length ({len(rho)}) != n_phases ({n_phases})")
        if len(mu) != n_phases:
            raise ValueError(f"mu length ({len(mu)}) != n_phases ({n_phases})")
        for i, r in enumerate(rho):
            if r <= 0:
                raise ValueError(f"rho[{i}] must be positive, got {r}")
        for i, m in enumerate(mu):
            if m <= 0:
                raise ValueError(f"mu[{i}] must be positive, got {m}")

        self._n_phases = n_phases
        self._phase_names = list(phase_names)
        self._rho = list(rho)
        self._mu = list(mu)
        self._C_alpha = C_alpha

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_phases(self) -> int:
        """Number of phases."""
        return self._n_phases

    @property
    def phase_names(self) -> list[str]:
        """Phase names."""
        return self._phase_names.copy()

    @property
    def rho(self) -> list[float]:
        """Phase densities (kg/m^3)."""
        return self._rho.copy()

    @property
    def mu(self) -> list[float]:
        """Phase viscosities (Pa·s)."""
        return self._mu.copy()

    @property
    def C_alpha(self) -> float:
        """Compression coefficient."""
        return self._C_alpha

    # ------------------------------------------------------------------
    # Volume fraction constraint
    # ------------------------------------------------------------------

    def compute_last_alpha(
        self, alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Compute N-th phase volume fraction from constraint.

        alpha_N = 1 - sum(alpha_1 .. alpha_{N-1}), clamped to [0, 1].

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` volume fractions of first N-1 phases.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` volume fraction of the N-th phase.
        """
        alpha_N = 1.0 - alphas.sum(dim=-1)
        return alpha_N.clamp(0.0, 1.0)

    def validate_alphas(self, alphas: torch.Tensor) -> torch.Tensor:
        """Validate and clamp volume fractions.

        Ensures each alpha in [0, 1] and renormalises to satisfy
        the summation constraint.

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` independent volume fractions.

        Returns
        -------
        torch.Tensor
            Clamped and renormalised volume fractions.
        """
        # Clamp each to [0, 1]
        alphas = alphas.clamp(0.0, 1.0)

        # Compute total and renormalise if needed
        total = alphas.sum(dim=-1) + 1e-30  # avoid /0
        # If total > 1, renormalise proportionally
        scale = torch.where(total > 1.0, 1.0 / total, torch.ones_like(total))
        alphas = alphas * scale.unsqueeze(-1)

        return alphas

    # ------------------------------------------------------------------
    # Mixture properties
    # ------------------------------------------------------------------

    def mixture_density(self, alphas: torch.Tensor) -> torch.Tensor:
        """Compute mixture density: rho_m = sum(alpha_i * rho_i).

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` independent volume fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture density (kg/m^3).
        """
        alphas = self.validate_alphas(alphas)
        device = alphas.device
        dtype = alphas.dtype

        rho_m = torch.zeros(alphas.shape[0], device=device, dtype=dtype)
        for i in range(self._n_phases - 1):
            rho_m = rho_m + alphas[:, i] * self._rho[i]
        # Add N-th phase contribution
        alpha_N = self.compute_last_alpha(alphas)
        rho_m = rho_m + alpha_N * self._rho[-1]

        return rho_m

    def mixture_viscosity(self, alphas: torch.Tensor) -> torch.Tensor:
        """Compute mixture viscosity: mu_m = sum(alpha_i * mu_i).

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` independent volume fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture dynamic viscosity (Pa·s).
        """
        alphas = self.validate_alphas(alphas)
        device = alphas.device
        dtype = alphas.dtype

        mu_m = torch.zeros(alphas.shape[0], device=device, dtype=dtype)
        for i in range(self._n_phases - 1):
            mu_m = mu_m + alphas[:, i] * self._mu[i]
        alpha_N = self.compute_last_alpha(alphas)
        mu_m = mu_m + alpha_N * self._mu[-1]

        return mu_m

    def mixture_kinematic_viscosity(
        self, alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixture kinematic viscosity: nu_m = mu_m / rho_m.

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` independent volume fractions.

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` mixture kinematic viscosity (m^2/s).
        """
        return self.mixture_viscosity(alphas) / self.mixture_density(alphas).clamp(min=1e-30)

    # ------------------------------------------------------------------
    # Advection
    # ------------------------------------------------------------------

    def advance_phase(
        self,
        alpha_i: torch.Tensor,
        phi: torch.Tensor,
        mesh: Any,
        delta_t: float,
    ) -> torch.Tensor:
        """Advance a single phase volume fraction by one time step.

        Uses upwind advection with interface compression and clamping.

        Parameters
        ----------
        alpha_i : torch.Tensor
            ``(n_cells,)`` volume fraction of one phase.
        phi : torch.Tensor
            ``(n_faces,)`` face flux.
        mesh : Any
            Finite volume mesh.
        delta_t : float
            Time step (s).

        Returns
        -------
        torch.Tensor
            ``(n_cells,)`` updated volume fraction.
        """
        device = alpha_i.device
        dtype = alpha_i.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour
        cell_volumes = mesh.cell_volumes

        int_owner = owner[:n_internal]
        int_neigh = neighbour

        # Upwind interpolation
        flux = phi[:n_internal]
        is_positive = flux >= 0.0
        alpha_P = gather(alpha_i, int_owner)
        alpha_N = gather(alpha_i, int_neigh)
        alpha_face = torch.where(is_positive, alpha_P, alpha_N)

        # Compression flux
        phi_max = flux.abs().max().clamp(min=1e-30)
        delta_alpha = alpha_P - alpha_N
        compression_flux = self._C_alpha * phi_max * delta_alpha

        # Total flux
        alpha_flux = flux * alpha_face + compression_flux

        # Divergence
        div_alpha = torch.zeros(n_cells, dtype=dtype, device=device)
        div_alpha = div_alpha + scatter_add(alpha_flux, int_owner, n_cells)
        div_alpha = div_alpha + scatter_add(-alpha_flux, int_neigh, n_cells)

        # Boundary faces
        if mesh.n_faces > n_internal:
            bnd_flux = phi[n_internal:] * gather(alpha_i, owner[n_internal:])
            div_alpha = div_alpha + scatter_add(bnd_flux, owner[n_internal:], n_cells)

        # Forward Euler + clamp
        V = cell_volumes.clamp(min=1e-30)
        alpha_new = alpha_i - delta_t * div_alpha / V
        alpha_new = alpha_new.clamp(0.0, 1.0)

        return alpha_new

    def advance(
        self,
        alphas: torch.Tensor,
        phi: torch.Tensor,
        mesh: Any,
        delta_t: float,
    ) -> torch.Tensor:
        """Advance all independent volume fractions by one time step.

        Parameters
        ----------
        alphas : torch.Tensor
            ``(n_cells, N-1)`` independent volume fractions.
        phi : torch.Tensor
            ``(n_faces,)`` face flux.
        mesh : Any
            Finite volume mesh.
        delta_t : float
            Time step (s).

        Returns
        -------
        torch.Tensor
            ``(n_cells, N-1)`` updated independent volume fractions.
        """
        alphas = self.validate_alphas(alphas)
        updated = []
        for i in range(self._n_phases - 1):
            alpha_new_i = self.advance_phase(alphas[:, i], phi, mesh, delta_t)
            updated.append(alpha_new_i)

        result = torch.stack(updated, dim=-1)
        return self.validate_alphas(result)

    def __repr__(self) -> str:
        phases = ", ".join(self._phase_names)
        return (
            f"IncompressibleMultiphaseVoF(n_phases={self._n_phases}, "
            f"phases=[{phases}], C_alpha={self._C_alpha})"
        )
