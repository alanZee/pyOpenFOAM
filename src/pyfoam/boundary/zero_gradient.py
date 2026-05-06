"""
zeroGradient boundary condition.

Implements ∂φ/∂n = 0 at the boundary face.  In OpenFOAM syntax::

    type   zeroGradient;

No matrix contribution — the boundary flux is zero by construction.
The boundary face value equals the adjacent cell value.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ZeroGradientBC"]


@BoundaryCondition.register("zeroGradient")
class ZeroGradientBC(BoundaryCondition):
    """Zero-gradient (Neumann) boundary condition.

    The normal gradient at the boundary face is zero, meaning the
    boundary value equals the adjacent internal-cell value.
    """

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Copy adjacent cell values to boundary faces.

        For a cell-centred field, the boundary-face value is the
        value of the owner cell.
        """
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]
        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = owner_values
        else:
            field[self._patch.face_indices] = owner_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero gradient: no matrix contribution."""
        device = get_device()
        dtype = field.dtype
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source
