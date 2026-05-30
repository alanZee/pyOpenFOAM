"""Enhanced scaled heat flux boundary condition (v9).

In OpenFOAM syntax::

    type        scaledHeatFlux9;
    value       uniform 0;

Coefficients:
    - Standard scaled heat flux parameters (from base and earlier versions).
    - ``spatial_period_coeff`` (float): Spatial periodicity correction. (default 0.0).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ScaledHeatFlux9BC"]


@BoundaryCondition.register("scaledHeatFlux9")
class ScaledHeatFlux9BC(BoundaryCondition):
    """Enhanced scaled heat flux v9.

    - ``spatial_period_coeff`` (float): Spatial periodicity correction. (default 0.0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._history_coeff = float(self._coeffs.get("history_coeff", 0.1))
        self._spatial_period_coeff = float(self._coeffs.get("spatial_period_coeff", 0.0))

    @property
    def history_coeff(self) -> float:
        return self._history_coeff

    @property
    def spatial_period_coeff(self) -> float:
        return self._spatial_period_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced scaled heat flux v9."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v9 enhancement: spatial periodicity correction.
        x_norm = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        W_period = 1.0 + self._spatial_period_coeff * torch.cos(6.28318 * x_norm)
        values = values * W_period

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
        """Penalty method for v9 enhanced scaled heat flux BC."""
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
