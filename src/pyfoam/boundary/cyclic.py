"""
cyclic boundary condition.

Connects two matching patches so that values from one side are used
on the other.  In OpenFOAM syntax::

    type            cyclic;
    neighbourPatch  cyclic_half1;

Cyclic BCs enforce face-value equality between coupled patches.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["CyclicBC"]


@BoundaryCondition.register("cyclic")
class CyclicBC(BoundaryCondition):
    """Cyclic (periodic) boundary condition.

    Pairs two patches so that boundary face values on one side equal
    the values on the other side.

    The ``neighbour_patch`` field in :class:`Patch` must be set to
    the name of the coupled patch.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._neighbour_field: torch.Tensor | None = None

    def set_neighbour_field(self, neighbour_field: torch.Tensor) -> None:
        """Set the neighbour-patch face values for coupling.

        Args:
            neighbour_field: Tensor of face values from the coupled patch.
        """
        self._neighbour_field = neighbour_field.to(
            dtype=get_default_dtype(), device=get_device()
        )

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Copy neighbour patch values to this patch's faces.

        If no neighbour field has been explicitly set, copies from the
        owner cells (degenerate zero-gradient behaviour).
        """
        if self._neighbour_field is not None:
            values = self._neighbour_field
        else:
            # Fallback: copy from owner cells
            owners = self._patch.owner_cells.to(device=field.device)
            values = field[owners]

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = values
        else:
            field[self._patch.face_indices] = values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cyclic coupling: implicit diagonal + source from neighbour.

        For each face on this patch:
            diag[c]   += deltaCoeff * area
            source[c] += deltaCoeff * area * neighbourValue

        This mirrors the fixedValue penalty approach but with the
        value taken from the coupled patch.
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

        if self._neighbour_field is not None:
            nvalues = self._neighbour_field.to(device=device, dtype=dtype)
            source.scatter_add_(0, owners, coeff * nvalues)
        else:
            # No neighbour data → treat as zero-flux
            pass

        return diag, source
