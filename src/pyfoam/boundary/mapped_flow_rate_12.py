"""Enhanced mapped flow rate boundary condition (v12).

In OpenFOAM syntax::

    type        mappedFlowRate12;
    value       uniform 0;

Coefficients:
    - Standard mapped flow rate parameters (from base and earlier versions).
    - ``turb_disp_coeff`` (float): Turbulent dispersion correction. (default 0.05).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedFlowRate12BC"]


@BoundaryCondition.register("mappedFlowRate12")
class MappedFlowRate12BC(BoundaryCondition):
    """Enhanced mapped flow rate v12.

    - ``turb_disp_coeff`` (float): Turbulent dispersion correction. (default 0.05).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._shear_disp_coeff = float(self._coeffs.get("shear_disp_coeff", 0.1))
        self._turb_disp_coeff = float(self._coeffs.get("turb_disp_coeff", 0.05))

    @property
    def shear_disp_coeff(self) -> float:
        return self._shear_disp_coeff

    @property
    def turb_disp_coeff(self) -> float:
        return self._turb_disp_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced mapped flow rate v12."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v12 enhancement: turbulent dispersion correction.
        r_frac = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        disp_t = self._turb_disp_coeff * torch.sqrt(torch.clamp(r_frac, min=1e-30))
        values = values * (1.0 + disp_t)

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
        """Penalty method for v12 enhanced mapped flow rate BC."""
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
