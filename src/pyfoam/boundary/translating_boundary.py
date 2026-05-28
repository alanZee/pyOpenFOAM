"""
Translating boundary condition for moving reference frame simulations.

Applies a uniform translation velocity to a boundary patch, shifting
the velocity field by a specified translation vector:

    U_boundary = U_internal + U_translate

This is used in OpenFOAM's ``translatingBoundary`` type for simulating
moving walls or translating reference frames (e.g., conveyor belts,
linear slides, or translating domains).

Usage::

    type            translatingBoundary;
    U_translate     (1 0 0);     // translation velocity (m/s)
    value           uniform (0 0 0);
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TranslatingBoundaryBC"]

logger = logging.getLogger(__name__)


@BoundaryCondition.register("translatingBoundary")
class TranslatingBoundaryBC(BoundaryCondition):
    """Translating boundary condition for moving reference frames.

    Sets the boundary velocity to the internal cell velocity plus a
    constant translation vector:

        U_boundary = U_internal + U_translate

    This allows the entire domain or a boundary to "move" with a
    specified velocity, which is subtracted from the solution in the
    moving reference frame.

    Coefficients:
        - ``U_translate``: Translation velocity vector (m/s),
          default (0, 0, 0).
        - ``value``: Initial velocity (overwritten on apply).
    """

    def __init__(
        self,
        patch: Patch,
        coeffs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(patch, coeffs)
        self._U_translate = self._parse_vector(
            self._coeffs.get("U_translate", [0.0, 0.0, 0.0])
        )

    @staticmethod
    def _parse_vector(value: Any) -> torch.Tensor:
        """Parse a vector value from coefficients."""
        if isinstance(value, torch.Tensor):
            return value.to(dtype=get_default_dtype(), device="cpu")
        if isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=get_default_dtype())
        # String format: "(x y z)"
        if isinstance(value, str):
            parts = value.strip("() ").split()
            return torch.tensor([float(v) for v in parts], dtype=get_default_dtype())
        return torch.zeros(3, dtype=get_default_dtype())

    @property
    def U_translate(self) -> torch.Tensor:
        """Translation velocity vector ``(3,)``."""
        return self._U_translate

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        U_internal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply translating boundary: U = U_internal + U_translate.

        Args:
            field: Velocity field ``(n_total, 3)``.
            patch_idx: Optional contiguous start index.
            U_internal: Internal cell velocity adjacent to each face
                        ``(n_faces, 3)``. Defaults to zero.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        U_t = self._U_translate.to(device=device, dtype=dtype)

        if U_internal is None:
            U_internal = torch.zeros(n, 3, dtype=dtype, device=device)
        else:
            U_internal = U_internal.to(device=device, dtype=dtype)

        # U_boundary = U_internal + U_translate
        U_boundary = U_internal + U_t.unsqueeze(0)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = U_boundary
        else:
            field[self._patch.face_indices] = U_boundary

        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        U_internal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method: diagonal += deltaCoeff * area, source += coeff * U_x."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        n = self._patch.n_faces
        U_t = self._U_translate.to(device=device, dtype=dtype)

        if U_internal is None:
            U_internal = torch.zeros(n, 3, dtype=dtype, device=device)
        else:
            U_internal = U_internal.to(device=device, dtype=dtype)

        U_boundary = U_internal + U_t.unsqueeze(0)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        # Source: x-component of velocity (standard for vector BCs)
        source.scatter_add_(0, owners, coeff * U_boundary[:, 0])

        return diag, source
