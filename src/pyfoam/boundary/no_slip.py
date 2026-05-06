"""
noSlip boundary condition.

Sets velocity to zero at wall faces.  In OpenFOAM syntax::

    type   noSlip;

Implemented as a fixedValue BC with value ``(0, 0, 0)``.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["NoSlipBC"]


@BoundaryCondition.register("noSlip")
class NoSlipBC(BoundaryCondition):
    """No-slip wall boundary condition.

    Velocity is fixed to zero at all boundary faces.
    For vector fields, the zero vector is prescribed.
    For scalar fields, zero is prescribed (useful as a fallback).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face values to zero."""
        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = 0.0
        else:
            field[self._patch.face_indices] = 0.0
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method with prescribed value = 0.

        Same as fixedValue(0): large diagonal coefficient, zero source.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        # Source is zero because prescribed value = 0

        return diag, source
