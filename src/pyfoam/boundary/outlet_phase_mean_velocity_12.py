"""Enhanced outlet phase mean velocity boundary condition (v12).

In OpenFOAM syntax::

    type        outletPhaseMeanVelocity12;
    value       uniform 0;

Coefficients:
    - Standard outlet phase mean velocity parameters (from base and earlier versions).
    - ``wall_pressure_coeff`` (float): Wall-pressure correction at outlet. (default 0.03).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["OutletPhaseMeanVelocity12BC"]


@BoundaryCondition.register("outletPhaseMeanVelocity12")
class OutletPhaseMeanVelocity12BC(BoundaryCondition):
    """Enhanced outlet phase mean velocity v12.

    - ``wall_pressure_coeff`` (float): Wall-pressure correction at outlet. (default 0.03).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._axial_decay_coeff = float(self._coeffs.get("axial_decay_coeff", 0.1))
        self._turb_disp_coeff = float(self._coeffs.get("turb_disp_coeff", 0.05))
        self._radial_balance_coeff = float(self._coeffs.get("radial_balance_coeff", 0.02))
        self._entropy_coeff = float(self._coeffs.get("entropy_coeff", 0.01))
        self._wall_pressure_coeff = float(self._coeffs.get("wall_pressure_coeff", 0.03))

    @property
    def axial_decay_coeff(self) -> float:
        return self._axial_decay_coeff

    @property
    def turb_disp_coeff(self) -> float:
        return self._turb_disp_coeff

    @property
    def radial_balance_coeff(self) -> float:
        return self._radial_balance_coeff

    @property
    def entropy_coeff(self) -> float:
        return self._entropy_coeff

    @property
    def wall_pressure_coeff(self) -> float:
        return self._wall_pressure_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced outlet phase mean velocity v12."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v12 enhancement: wall-pressure correction at outlet.
        r_frac = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        vel_mag = values.norm(dim=-1) if values.dim() > 1 else values.abs()
        values = values * (1.0 + self._wall_pressure_coeff * r_frac.unsqueeze(-1) * 0.01) if values.dim() > 1 else values

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
        """Penalty method for v12 enhanced outlet phase mean velocity BC."""
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
