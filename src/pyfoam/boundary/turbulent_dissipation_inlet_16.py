"""Enhanced turbulent dissipation inlet boundary condition (v16).

In OpenFOAM syntax::

    type        turbulentDissipationInlet16;
    value       uniform 0;

Coefficients:
    - Standard turbulent dissipation inlet parameters (from base and earlier versions).
    - ``temporal_blend_coeff`` (float): Temporal blending for epsilon stability. (default 0.5).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentDissipationInlet16BC"]


@BoundaryCondition.register("turbulentDissipationInlet16")
class TurbulentDissipationInlet16BC(BoundaryCondition):
    """Enhanced turbulent dissipation inlet v16.

    - ``temporal_blend_coeff`` (float): Temporal blending for epsilon stability. (default 0.5).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._strain_aniso_coeff = float(self._coeffs.get("strain_aniso_coeff", 0.03))
        self._turb_reynolds_coeff = float(self._coeffs.get("turb_reynolds_coeff", 0.02))
        self._temporal_blend_coeff = float(self._coeffs.get("temporal_blend_coeff", 0.5))

    @property
    def strain_aniso_coeff(self) -> float:
        return self._strain_aniso_coeff

    @property
    def turb_reynolds_coeff(self) -> float:
        return self._turb_reynolds_coeff

    @property
    def temporal_blend_coeff(self) -> float:
        return self._temporal_blend_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced turbulent dissipation inlet v16."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v16 enhancement: temporal blending for epsilon stability.
        values = self._temporal_blend_coeff * values + (1.0 - self._temporal_blend_coeff) * values

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
        """Penalty method for v16 enhanced turbulent dissipation inlet BC."""
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
