"""Enhanced slip wall boundary condition (v3).

In OpenFOAM syntax::

    type        slip3;
    value       uniform 0;

Coefficients:
    - Standard slip wall parameters (from base and earlier versions).
    - ``normal_correction`` (float): Normal component residual correction. (default 0.0).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["Slip3BC"]


@BoundaryCondition.register("slip3")
class Slip3BC(BoundaryCondition):
    """Enhanced slip wall v3.

    - ``normal_correction`` (float): Normal component residual correction. (default 0.0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._tangential_correction = float(self._coeffs.get("tangential_correction", 0.0))
        self._normal_correction = float(self._coeffs.get("normal_correction", 0.0))

    @property
    def tangential_correction(self) -> float:
        return self._tangential_correction

    @property
    def normal_correction(self) -> float:
        return self._normal_correction
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced slip wall v3."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v3 enhancement: normal component residual correction.
        values = values * (1.0 + self._normal_correction)

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
        """Penalty method for v3 enhanced slip wall BC."""
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
