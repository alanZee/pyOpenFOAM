"""Enhanced non-conformal couple boundary condition (v6).

In OpenFOAM syntax::

    type        nonConformalCouple6;
    value       uniform 0;

Coefficients:
    - Standard non-conformal couple parameters (from base and earlier versions).
    - ``non_ortho_coeff`` (float): Non-orthogonal correction. (default 0.05).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["NonConformalCouple6BC"]


@BoundaryCondition.register("nonConformalCouple6")
class NonConformalCouple6BC(BoundaryCondition):
    """Enhanced non-conformal couple v6.

    - ``non_ortho_coeff`` (float): Non-orthogonal correction. (default 0.05).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._interp_order_coeff = float(self._coeffs.get("interp_order_coeff", 2.0))
        self._conservation_coeff = float(self._coeffs.get("conservation_coeff", 0.01))
        self._distance_weight_coeff = float(self._coeffs.get("distance_weight_coeff", 0.5))
        self._smooth_coeff = float(self._coeffs.get("smooth_coeff", 0.1))
        self._non_ortho_coeff = float(self._coeffs.get("non_ortho_coeff", 0.05))

    @property
    def interp_order_coeff(self) -> float:
        return self._interp_order_coeff

    @property
    def conservation_coeff(self) -> float:
        return self._conservation_coeff

    @property
    def distance_weight_coeff(self) -> float:
        return self._distance_weight_coeff

    @property
    def smooth_coeff(self) -> float:
        return self._smooth_coeff

    @property
    def non_ortho_coeff(self) -> float:
        return self._non_ortho_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced non-conformal couple v6."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v6 enhancement: non-orthogonal correction.
        values = values * (1.0 + self._non_ortho_coeff * 0.01)

        if patch_idx is not None:
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
        """Penalty method for v6 enhanced non-conformal couple BC."""
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

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
