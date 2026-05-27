"""
mapped boundary condition.

Copies field values from a mapped (neighbour) patch.
In OpenFOAM syntax::

    type            mapped;
    neighbourPatch  outlet;

The ``neighbourPatch`` coefficient names the patch from which face values
are copied.  If no mapped field is provided, falls back to zero-gradient
(owner cell value).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedBC"]


@BoundaryCondition.register("mapped")
class MappedBC(BoundaryCondition):
    """Mapped boundary condition.

    Copies face values from a neighbouring patch.  The coupled patch
    name is specified via ``neighbourPatch`` in coefficients.

    Usage::

        bc = MappedBC(patch, {"neighbourPatch": "otherPatch"})
        bc.set_mapped_field(other_face_values)
        bc.apply(field)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mapped_field: torch.Tensor | None = None

    @property
    def neighbour_patch_name(self) -> str | None:
        """Return the name of the mapped neighbour patch."""
        return self._coeffs.get("neighbourPatch", self._patch.neighbour_patch)

    def set_mapped_field(self, mapped_field: torch.Tensor) -> None:
        """Set face values from the mapped neighbour patch.

        Args:
            mapped_field: Tensor of face values from the coupled patch.
        """
        self._mapped_field = mapped_field.to(
            dtype=get_default_dtype(), device=get_device()
        )

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Copy mapped patch values to this patch's faces.

        If no mapped field has been set, falls back to zero-gradient
        (owner cell values).
        """
        if self._mapped_field is not None:
            values = self._mapped_field.to(device=field.device, dtype=field.dtype)
        else:
            # Fallback: zero-gradient (copy from owner cells)
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
        """Implicit diagonal + source from mapped patch values."""
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

        if self._mapped_field is not None:
            values = self._mapped_field.to(device=device, dtype=dtype)
            source.scatter_add_(0, owners, coeff * values)
        # else: no mapped data -> treat as zero-flux (no source contribution)

        return diag, source
