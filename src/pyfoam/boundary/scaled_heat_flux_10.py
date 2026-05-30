"""Enhanced scaled heat flux boundary condition (v10).

In OpenFOAM syntax::

    type        scaledHeatFlux10;
    value       uniform 0;

Coefficients:
    - Standard scaled heat flux parameters (from base and earlier versions).
    - ``contact_coeff`` (float): Contact resistance correction. (default 0.0).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ScaledHeatFlux10BC"]


@BoundaryCondition.register("scaledHeatFlux10")
class ScaledHeatFlux10BC(BoundaryCondition):
    """Enhanced scaled heat flux v10.

    - ``contact_coeff`` (float): Contact resistance correction. (default 0.0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._history_coeff = float(self._coeffs.get("history_coeff", 0.1))
        self._spatial_period_coeff = float(self._coeffs.get("spatial_period_coeff", 0.0))
        self._contact_coeff = float(self._coeffs.get("contact_coeff", 0.0))

    @property
    def history_coeff(self) -> float:
        return self._history_coeff

    @property
    def spatial_period_coeff(self) -> float:
        return self._spatial_period_coeff

    @property
    def contact_coeff(self) -> float:
        return self._contact_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced scaled heat flux v10."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v10 enhancement: contact resistance correction.
        values = values * (1.0 + self._contact_coeff * 0.01)

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
        """Penalty method for v10 enhanced scaled heat flux BC."""
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
