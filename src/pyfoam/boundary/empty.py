"""
empty boundary condition.

Used for 2D and 1D simulations to denote patch faces that have no
physical area in the solved dimension.  In OpenFOAM syntax::

    type   empty;

The ``apply()`` method is a no-op (field unchanged).
Matrix contributions are zero — the patch adds nothing to the
linear system.
"""

from __future__ import annotations

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition

__all__ = ["EmptyBC"]


@BoundaryCondition.register("empty")
class EmptyBC(BoundaryCondition):
    """Empty boundary condition for 2D / 1D cases.

    Faces on an ``empty`` patch do not participate in the calculation.
    Both ``apply()`` and ``matrix_contributions()`` are identity
    operations.
    """

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """No-op: field is returned unchanged."""
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero matrix contribution."""
        device = get_device()
        dtype = field.dtype
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source
