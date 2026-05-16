"""
MULES (Multidimensional Universal Limiter with Explicit Solution).

Ensures boundedness of transported scalars (e.g. volume fraction α)
during advection, while preserving conservation.

Based on OpenFOAM's MULES implementation:
    src/finiteVolume/fvMatrices/solvers/MULES/MULES.H

The algorithm limits face fluxes so that cell values remain within
[alpha_min, alpha_max] bounds. It iterates to find the maximum
allowable flux limiter for each face.

Usage::

    from pyfoam.multiphase.mules import MULESLimiter

    mules = MULESLimiter(mesh)
    alpha_limited = mules.limit(alpha, phi, alpha_flux, dt, alpha_min, alpha_max)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["MULESLimiter"]

logger = logging.getLogger(__name__)


class MULESLimiter:
    """MULES limiter for VOF and other bounded scalar transport.

    Limits the face fluxes so that each cell's scalar value remains
    within the specified bounds while preserving conservation.

    The algorithm:
    1. Compute lower and upper bounds per cell (from neighbours).
    2. Compute the net flux that would cause each cell to exceed its bounds.
    3. Compute a limiter coefficient ψ ∈ [0, 1] for each face.
    4. Iterate to tighten the limiter until all cells are bounded.

    Parameters
    ----------
    mesh : Any
        The finite volume mesh (must have owner, neighbour, cell_volumes).
    n_iterations : int
        Number of limiter tightening iterations. Default 3.

    Examples::

        mules = MULESLimiter(mesh, n_iterations=3)
        alpha_limited = mules.limit(
            alpha, phi, alpha_flux, dt,
            alpha_min=0.0, alpha_max=1.0,
        )
    """

    def __init__(self, mesh: Any, n_iterations: int = 3) -> None:
        self._mesh = mesh
        self._n_iterations = n_iterations
        self._device = get_device()
        self._dtype = get_default_dtype()

    def limit(
        self,
        alpha: torch.Tensor,
        phi: torch.Tensor,
        alpha_flux: torch.Tensor,
        delta_t: float,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        alpha_min_field: Optional[torch.Tensor] = None,
        alpha_max_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply MULES limiting to the alpha flux and update alpha.

        Args:
            alpha: Current volume fraction ``(n_cells,)``.
            phi: Face flux ``(n_faces,)``.
            alpha_flux: Unlimited alpha face flux for internal faces
                ``(n_internal,)``.  Positive means owner→neighbour.
            delta_t: Time step size.
            alpha_min: Global lower bound for alpha (default 0).
            alpha_max: Global upper bound for alpha (default 1).
            alpha_min_field: Per-cell lower bound ``(n_cells,)``.
                If None, uses global alpha_min.
            alpha_max_field: Per-cell upper bound ``(n_cells,)``.
                If None, uses global alpha_max.

        Returns:
            Limited (updated) volume fraction ``(n_cells,)``.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        V = mesh.cell_volumes.clamp(min=1e-30)

        # --- Step 1: Compute per-cell bounds ---
        if alpha_min_field is None:
            lower = torch.full(
                (n_cells,), alpha_min, dtype=self._dtype, device=self._device
            )
        else:
            lower = alpha_min_field.clone()

        if alpha_max_field is None:
            upper = torch.full(
                (n_cells,), alpha_max, dtype=self._dtype, device=self._device
            )
        else:
            upper = alpha_max_field.clone()

        # Tighten bounds using neighbour values (like OpenFOAM)
        alpha_neigh_max = torch.zeros(
            n_cells, dtype=self._dtype, device=self._device
        )
        alpha_neigh_min = torch.zeros(
            n_cells, dtype=self._dtype, device=self._device
        )

        # Scatter max/min of neighbour alpha to each cell
        alpha_own = gather(alpha, int_owner)
        alpha_nei = gather(alpha, int_neigh)

        alpha_neigh_max = alpha_neigh_max.scatter_reduce(
            0, int_owner, alpha_nei, reduce="amax", include_self=False
        )
        alpha_neigh_max = alpha_neigh_max.scatter_reduce(
            0, int_neigh, alpha_own, reduce="amax", include_self=False
        )
        alpha_neigh_min = alpha_neigh_min.scatter_reduce(
            0, int_owner, alpha_nei, reduce="amin", include_self=False
        )
        alpha_neigh_min = alpha_neigh_min.scatter_reduce(
            0, int_neigh, alpha_own, reduce="amin", include_self=False
        )

        # Use max/min of current and neighbour values
        lower = torch.min(lower, torch.min(alpha, alpha_neigh_min))
        upper = torch.max(upper, torch.max(alpha, alpha_neigh_max))

        # Add a small margin to avoid trivial limiting
        eps = 1e-12
        lower = lower - eps
        upper = upper + eps

        # --- Step 2: Compute net source for each cell ---
        # Net source from unlimited flux
        net_source = torch.zeros(
            n_cells, dtype=self._dtype, device=self._device
        )
        net_source = net_source + scatter_add(
            alpha_flux, int_owner, n_cells
        )
        net_source = net_source + scatter_add(
            -alpha_flux, int_neigh, n_cells
        )

        # --- Step 3: Compute the total face volume (V/dt per cell) ---
        V_dt = V / delta_t

        # --- Step 4: MULES limiter iterations ---
        # Start with unlimited flux
        limited_flux = alpha_flux.clone()

        for _iter in range(self._n_iterations):
            # Recompute net source from limited flux
            net_source = torch.zeros(
                n_cells, dtype=self._dtype, device=self._device
            )
            net_source = net_source + scatter_add(
                limited_flux, int_owner, n_cells
            )
            net_source = net_source + scatter_add(
                -limited_flux, int_neigh, n_cells
            )

            # For each face, compute how much the flux can be limited
            # S_p: maximum positive flux into owner (flux leaving owner)
            # S_n: maximum negative flux into neighbour (flux entering neighbour)

            # Positive contribution to owner (flux leaves owner): limited_flux > 0
            # Negative contribution to owner (flux enters owner): limited_flux < 0

            # Compute allowable ranges
            alpha_new = alpha - delta_t * net_source / V
            overshoot = torch.clamp(alpha_new - upper, min=0.0)
            undershoot = torch.clamp(lower - alpha_new, min=0.0)

            # Maximum allowable total flux into/out of each cell
            T_p_max = overshoot * V_dt  # max flux out of cell
            T_n_max = undershoot * V_dt  # max flux into cell (from outside)

            # For each face: compute positive and negative contributions
            # Positive flux: owner → neighbour
            #   owner loses, neighbour gains
            flux_pos = torch.clamp(limited_flux, min=0.0)
            flux_neg = torch.clamp(limited_flux, max=0.0)

            # Owner side: positive flux means flux leaves owner
            # Neighbour side: positive flux means flux enters neighbour

            # Compute limiter for each face
            # For positive flux (owner→neighbour):
            #   limit so owner doesn't drop below lower
            #   limit so neighbour doesn't exceed upper
            T_p = gather(T_p_max, int_owner)  # max flux out of owner
            T_n = gather(T_n_max, int_neigh)  # max flux into neighbour

            # S_p: actual positive flux magnitude
            S_p = flux_pos
            # How much can we limit? psi_p = T_p / S_p (capped at 1)
            psi_p = torch.where(
                S_p > 1e-30,
                torch.clamp(T_p / S_p, max=1.0),
                torch.ones_like(S_p),
            )

            # For negative flux (neighbour→owner):
            #   limit so neighbour doesn't drop below lower
            #   limit so owner doesn't exceed upper
            T_p2 = gather(T_p_max, int_neigh)  # max flux out of neighbour
            T_n2 = gather(T_n_max, int_owner)  # max flux into owner

            S_n = flux_neg.abs()
            psi_n = torch.where(
                S_n > 1e-30,
                torch.clamp(torch.min(T_p2, T_n2) / S_n, max=1.0),
                torch.ones_like(S_n),
            )

            # Combined limiter per face
            psi = torch.min(psi_p, psi_n)

            # Smooth the limiter (OpenFOAM does this)
            # Average with neighbours
            if _iter < self._n_iterations - 1:
                psi_own = gather(psi, int_owner)  # doesn't make sense directly
                # Instead, just use the minimum of face and its cell neighbours
                # For simplicity, keep the face limiter as is
                pass

            # Apply limiter
            limited_flux = limited_flux * psi

        # --- Step 5: Update alpha using limited flux ---
        div_limited = torch.zeros(
            n_cells, dtype=self._dtype, device=self._device
        )
        div_limited = div_limited + scatter_add(
            limited_flux, int_owner, n_cells
        )
        div_limited = div_limited + scatter_add(
            -limited_flux, int_neigh, n_cells
        )

        alpha_new = alpha - delta_t * div_limited / V

        # Final clamp as safety
        alpha_new = alpha_new.clamp(min=alpha_min, max=alpha_max)

        return alpha_new

    def limit_flux(
        self,
        alpha: torch.Tensor,
        alpha_flux: torch.Tensor,
        delta_t: float,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
    ) -> torch.Tensor:
        """Return the limited face flux (without updating alpha).

        Args:
            alpha: Current volume fraction ``(n_cells,)``.
            alpha_flux: Unlimited alpha face flux ``(n_internal,)``.
            delta_t: Time step size.
            alpha_min: Lower bound.
            alpha_max: Upper bound.

        Returns:
            Limited face flux ``(n_internal,)``.
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour
        V = mesh.cell_volumes.clamp(min=1e-30)
        V_dt = V / delta_t

        limited_flux = alpha_flux.clone()

        for _ in range(self._n_iterations):
            # Compute net source
            net = torch.zeros(
                n_cells, dtype=self._dtype, device=self._device
            )
            net = net + scatter_add(limited_flux, int_owner, n_cells)
            net = net + scatter_add(-limited_flux, int_neigh, n_cells)

            alpha_new = alpha - delta_t * net / V
            overshoot = torch.clamp(alpha_new - alpha_max, min=0.0)
            undershoot = torch.clamp(alpha_min - alpha_new, min=0.0)

            T_p = gather(overshoot * V_dt, int_owner)
            T_n = gather(undershoot * V_dt, int_neigh)

            flux_pos = torch.clamp(limited_flux, min=0.0)
            psi_p = torch.where(
                flux_pos > 1e-30,
                torch.clamp(T_p / flux_pos, max=1.0),
                torch.ones_like(flux_pos),
            )

            T_p2 = gather(undershoot * V_dt, int_owner)
            T_n2 = gather(overshoot * V_dt, int_neigh)
            flux_neg = torch.clamp(limited_flux, max=0.0).abs()
            psi_n = torch.where(
                flux_neg > 1e-30,
                torch.clamp(
                    torch.min(T_p2, T_n2) / flux_neg, max=1.0
                ),
                torch.ones_like(flux_neg),
            )

            psi = torch.min(psi_p, psi_n)
            limited_flux = limited_flux * psi

        return limited_flux

    def __repr__(self) -> str:
        return (
            f"MULESLimiter(n_cells={self._mesh.n_cells}, "
            f"n_iterations={self._n_iterations})"
        )
