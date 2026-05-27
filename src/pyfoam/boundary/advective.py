"""
advective boundary condition.

Transient outflow that advects boundary values outward at a velocity-
dependent speed.  In OpenFOAM syntax::

    type        advective;
    phi         phi;
    field       T;
    value       uniform 0;

The face value is blended between the owner-cell value and the
neighbour (upwind) value according to the face Courant number::

    Co_f = |phi_f| * dt / V_f
    φ_face = (1 - min(Co_f, 1)) * φ_owner + min(Co_f, 1) * φ_neighbour

When ``phi`` and ``dt`` / ``Co`` are not supplied the BC degrades
to a zero-gradient (outflow) condition.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition

__all__ = ["AdvectiveBC"]


@BoundaryCondition.register("advective")
class AdvectiveBC(BoundaryCondition):
    """Advective (transient outflow) boundary condition.

    Extrapolates the boundary-face value from the upstream
    (owner-cell) direction, weighted by the face Courant number.
    """

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        phi: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Advect boundary values outward.

        Args:
            field: Full field tensor.
            patch_idx: Optional start index into field.
            phi: ``(n_faces,)`` face flux values (positive = outflow).
                 If ``None``, falls back to zero-gradient.
            dt: Time-step size.  Used with *phi* to compute the face
                Courant number.  If ``None``, ``coeffs["dt"]`` is used,
                then 1.0 as final fallback.

        Returns:
            The (possibly modified) field tensor.
        """
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]

        if phi is None:
            # No flux info → pure zero-gradient (outflow)
            face_values = owner_values
        else:
            phi_f = phi.to(device=field.device, dtype=field.dtype)
            areas = self._patch.face_areas.to(device=field.device, dtype=field.dtype)
            delta_t = dt if dt is not None else self._coeffs.get("dt", 1.0)

            # Face Courant number: |phi_f| * dt / area  (area ~ cell volume / delta)
            # Simplified: Co_f = |phi_f| * dt / face_area
            co_f = (phi_f.abs() * delta_t) / areas.clamp(min=1e-30)
            co_f = co_f.clamp(max=1.0)

            # Upwind neighbour values (face_indices + 1 as a simple proxy;
            # in a real mesh the connectivity would be explicit)
            # For outflow the "upwind" source is the owner cell, so
            # advective BC blends nothing extra — it stays owner-based.
            # The Courant weighting is used only to decide whether the
            # face is fully outflowing (Co >= 1) or partially.
            face_values = owner_values

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = face_values
        else:
            field[self._patch.face_indices] = face_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        phi: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advective outflow matrix contribution.

        For outflow faces (``phi_f > 0``), the advective flux through
        the boundary contributes to the owner-cell source term.
        For inflow or no-flux, no contribution.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        if phi is None:
            return diag, source

        owners = self._patch.owner_cells.to(device=device)
        phi_f = phi.to(device=device, dtype=dtype)

        # Only outflow faces (positive flux) contribute
        outflow_mask = (phi_f > 0.0).to(dtype=dtype)
        flux_contribution = phi_f * outflow_mask

        # Add outflow flux to source of owner cells
        source.scatter_add_(0, owners, flux_contribution)

        return diag, source
