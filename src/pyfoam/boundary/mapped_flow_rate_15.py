"""Enhanced mapped flow rate boundary condition (v15).

In OpenFOAM syntax::

    type        mappedFlowRate15;
    value       uniform 0;

Coefficients:
    - Standard mapped flow rate parameters (from base and earlier versions).
    - ``entropy_coeff`` (float): Entropy generation correction. (default 0.01).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedFlowRate15BC"]


@BoundaryCondition.register("mappedFlowRate15")
class MappedFlowRate15BC(BoundaryCondition):
    """Enhanced mapped flow rate v15.

    - ``entropy_coeff`` (float): Entropy generation correction. (default 0.01).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._shear_disp_coeff = float(self._coeffs.get("shear_disp_coeff", 0.1))
        self._turb_disp_coeff = float(self._coeffs.get("turb_disp_coeff", 0.05))
        self._axial_decay_coeff = float(self._coeffs.get("axial_decay_coeff", 0.3))
        self._radial_pressure_grad = float(self._coeffs.get("radial_pressure_grad", 0.02))
        self._entropy_coeff = float(self._coeffs.get("entropy_coeff", 0.01))

    @property
    def shear_disp_coeff(self) -> float:
        return self._shear_disp_coeff

    @property
    def turb_disp_coeff(self) -> float:
        return self._turb_disp_coeff

    @property
    def axial_decay_coeff(self) -> float:
        return self._axial_decay_coeff

    @property
    def radial_pressure_grad(self) -> float:
        return self._radial_pressure_grad

    @property
    def entropy_coeff(self) -> float:
        return self._entropy_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced mapped flow rate v15."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v15 enhancement: entropy generation correction.
        r_frac = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        S_gen = self._entropy_coeff * torch.log(1.0 + values.abs().mean())
        values = values * (1.0 - S_gen * r_frac)

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
        """Penalty method for v15 enhanced mapped flow rate BC."""
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
