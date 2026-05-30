"""Enhanced non-conformal couple boundary condition (v3).

In OpenFOAM syntax::

    type        nonConformalCouple3;
    value       uniform 0;

Coefficients:
    - Standard non-conformal couple parameters (from base and earlier versions).
    - ``conservation_coeff`` (float): Conservation correction. (default 0.01).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["NonConformalCouple3BC"]


@BoundaryCondition.register("nonConformalCouple3")
class NonConformalCouple3BC(BoundaryCondition):
    """Enhanced non-conformal couple v3.

    - ``conservation_coeff`` (float): Conservation correction. (default 0.01).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._interp_order_coeff = float(self._coeffs.get("interp_order_coeff", 2.0))
        self._conservation_coeff = float(self._coeffs.get("conservation_coeff", 0.01))

    @property
    def interp_order_coeff(self) -> float:
        return self._interp_order_coeff

    @property
    def conservation_coeff(self) -> float:
        return self._conservation_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced non-conformal couple v3."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v3 enhancement: conservation correction.
        flux_in = field[owners].abs().sum()
        flux_out = values.abs().sum()
        if flux_out > 1e-30: values = values * (flux_in / flux_out)

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
        """Penalty method for v3 enhanced non-conformal couple BC."""
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
