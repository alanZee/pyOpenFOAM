"""
Non-conformal couple boundary condition.

Maps fields between non-matching mesh interfaces where face centres on
the two sides do not coincide.  In OpenFOAM syntax::

    type            nonConformalCouple;
    neighbourPatch  nbrPatch;
    transform       none;

Unlike a conformal cyclic, the non-conformal couple performs interpolation
to transfer data across mismatched face grids.  The interpolation weights
are computed from the supplied face-centre mapping arrays.

Usage::

    bc = NonConformalCoupleBC(patch, {"neighbourPatch": "otherPatch"})
    bc.set_mapping(nbr_values, weights, nbr_indices)
    bc.apply(field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["NonConformalCoupleBC"]


@BoundaryCondition.register("nonConformalCouple")
class NonConformalCoupleBC(BoundaryCondition):
    """Non-conformal mesh coupling boundary condition.

    Maps fields between non-matching mesh interfaces by weighted
    interpolation.  Each face on this patch is mapped to one or more
    faces on the neighbour patch; the interpolated value is the
    weighted average of the neighbour face values.

    Coefficients:
        - ``neighbourPatch``: Name of the coupled patch.
        - ``transform``: Coordinate transformation type (default ``"none"``).
        - ``value``: Initial field values (used for shape).

    Usage::

        bc = NonConformalCoupleBC(patch, {"neighbourPatch": "other"})
        bc.set_mapping(nbr_face_values, weights, nbr_face_indices)
        bc.apply(field)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._nbr_values: torch.Tensor | None = None
        self._weights: torch.Tensor | None = None
        self._nbr_indices: torch.Tensor | None = None
        self._transform: str = self._coeffs.get("transform", "none")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def neighbour_patch_name(self) -> str | None:
        """Return the name of the coupled neighbour patch."""
        return self._coeffs.get("neighbourPatch", self._patch.neighbour_patch)

    @property
    def transform(self) -> str:
        """Return the coordinate transform type."""
        return self._transform

    # ------------------------------------------------------------------
    # Mapping management
    # ------------------------------------------------------------------

    def set_mapping(
        self,
        nbr_values: torch.Tensor,
        weights: torch.Tensor,
        nbr_indices: torch.Tensor,
    ) -> None:
        """Set the interpolation mapping from the neighbour patch.

        Args:
            nbr_values: ``(n_nbr_faces,)`` field values on the neighbour patch.
            weights: ``(n_faces,)`` interpolation weight for each face on
                this patch (single-neighbour simplification).  Values
                should be in [0, 1].
            nbr_indices: ``(n_faces,)`` index into *nbr_values* for
                each face on this patch.
        """
        device = get_device()
        dtype = get_default_dtype()
        self._nbr_values = nbr_values.to(device=device, dtype=dtype)
        self._weights = weights.to(device=device, dtype=dtype)
        self._nbr_indices = nbr_indices.to(device=device)

    # ------------------------------------------------------------------
    # BoundaryCondition interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply interpolated values from the non-conformal neighbour.

        If no mapping has been set, falls back to zero-gradient
        (owner cell values).
        """
        if self._nbr_values is not None and self._weights is not None:
            device = field.device
            dtype = field.dtype
            nbr_vals = self._nbr_values.to(device=device, dtype=dtype)
            w = self._weights.to(device=device, dtype=dtype)
            idx = self._nbr_indices.to(device=device)

            # Weighted interpolation: value = w * nbr_value + (1 - w) * owner_value
            owners = self._patch.owner_cells.to(device=device)
            owner_vals = field[owners]
            mapped_vals = nbr_vals[idx]
            values = w * mapped_vals + (1.0 - w) * owner_vals
        else:
            # Fallback: zero-gradient
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
        """Implicit diagonal + source from non-conformal coupled values.

        Uses the penalty method with interpolation weights:

        - diag[c]   += w * deltaCoeff * area
        - source[c] += w * deltaCoeff * area * nbr_value
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

        if self._weights is not None and self._nbr_values is not None:
            w = self._weights.to(device=device, dtype=dtype)
            idx = self._nbr_indices.to(device=device)
            nbr_vals = self._nbr_values.to(device=device, dtype=dtype)

            # Apply weight to penalty coefficient
            weighted_coeff = w * coeff
            diag.scatter_add_(0, owners, weighted_coeff)
            source.scatter_add_(0, owners, weighted_coeff * nbr_vals[idx])
        else:
            # No mapping data -> treat as zero-flux
            diag.scatter_add_(0, owners, coeff)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
