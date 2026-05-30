"""Enhanced coupled temperature boundary condition (v3).

In OpenFOAM syntax::

    type        coupledTemperature3;
    value       uniform 0;

Coefficients:
    - Standard coupled temperature parameters (from base and earlier versions).
    - ``thermal_conductivity_ratio`` (float): Thermal conductivity ratio correction. (default 1.0).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["CoupledTemperature3BC"]


@BoundaryCondition.register("coupledTemperature3")
class CoupledTemperature3BC(BoundaryCondition):
    """Enhanced coupled temperature v3.

    - ``thermal_conductivity_ratio`` (float): Thermal conductivity ratio correction. (default 1.0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._contact_resistance_coeff = float(self._coeffs.get("contact_resistance_coeff", 0.0))
        self._thermal_conductivity_ratio = float(self._coeffs.get("thermal_conductivity_ratio", 1.0))

    @property
    def contact_resistance_coeff(self) -> float:
        return self._contact_resistance_coeff

    @property
    def thermal_conductivity_ratio(self) -> float:
        return self._thermal_conductivity_ratio
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced coupled temperature v3."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v3 enhancement: thermal conductivity ratio correction.
        values = values * self._thermal_conductivity_ratio

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
        """Penalty method for v3 enhanced coupled temperature BC."""
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
