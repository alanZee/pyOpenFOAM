"""
calculated boundary condition.

A boundary condition whose face values are derived from the
adjacent owner cell values.  In OpenFOAM syntax::

    type   calculated;

The ``apply()`` copies owner cell values to boundary faces
(same as zeroGradient).  The ``matrix_contributions()`` is zero,
reflecting that this BC does not independently drive the solution.
"""

from __future__ import annotations

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition

__all__ = ["CalculatedBC"]


@BoundaryCondition.register("calculated")
class CalculatedBC(BoundaryCondition):
    """Calculated field boundary condition.

    Face values are copied from the adjacent owner cells.
    This BC is typically used for fields that are computed
    (e.g. turbulence quantities) rather than independently solved.
    """

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Copy owner cell values to boundary faces."""
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
        """Zero matrix contribution.

        Calculated BC does not independently drive the solution;
        it only mirrors owner cell values to the boundary.
        """
        device = get_device()
        dtype = field.dtype
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source
