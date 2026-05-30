"""Enhanced turbulent dissipation inlet boundary condition (v15).

In OpenFOAM syntax::

    type        turbulentDissipationInlet15;
    value       uniform 0;

Coefficients:
    - Standard turbulent dissipation inlet parameters (from base and earlier versions).
    - ``turb_reynolds_coeff`` (float): Turbulent Reynolds number correction. (default 0.02).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentDissipationInlet15BC"]


@BoundaryCondition.register("turbulentDissipationInlet15")
class TurbulentDissipationInlet15BC(BoundaryCondition):
    """Enhanced turbulent dissipation inlet v15.

    - ``turb_reynolds_coeff`` (float): Turbulent Reynolds number correction. (default 0.02).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._strain_aniso_coeff = float(self._coeffs.get("strain_aniso_coeff", 0.03))
        self._turb_reynolds_coeff = float(self._coeffs.get("turb_reynolds_coeff", 0.02))

    @property
    def strain_aniso_coeff(self) -> float:
        return self._strain_aniso_coeff

    @property
    def turb_reynolds_coeff(self) -> float:
        return self._turb_reynolds_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced turbulent dissipation inlet v15."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v15 enhancement: turbulent reynolds number correction.
        values = values * (1.0 + self._turb_reynolds_coeff * 0.01)

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
        """Penalty method for v15 enhanced turbulent dissipation inlet BC."""
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
