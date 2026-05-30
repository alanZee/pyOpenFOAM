"""Enhanced outlet phase mean velocity boundary condition (v8).

In OpenFOAM syntax::

    type        outletPhaseMeanVelocity8;
    value       uniform 0;

Coefficients:
    - Standard outlet phase mean velocity parameters (from base and earlier versions).
    - ``axial_decay_coeff`` (float): Axial velocity decay at outlet. (default 0.1).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["OutletPhaseMeanVelocity8BC"]


@BoundaryCondition.register("outletPhaseMeanVelocity8")
class OutletPhaseMeanVelocity8BC(BoundaryCondition):
    """Enhanced outlet phase mean velocity v8.

    - ``axial_decay_coeff`` (float): Axial velocity decay at outlet. (default 0.1).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._axial_decay_coeff = float(self._coeffs.get("axial_decay_coeff", 0.1))

    @property
    def axial_decay_coeff(self) -> float:
        return self._axial_decay_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced outlet phase mean velocity v8."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v8 enhancement: axial velocity decay at outlet.
        r_frac = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        decay = torch.exp(-self._axial_decay_coeff * r_frac)
        values = values * decay.unsqueeze(-1) if values.dim() > 1 else values * decay

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
        """Penalty method for v8 enhanced outlet phase mean velocity BC."""
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
