"""Enhanced turbulent length scale inlet boundary condition (v14).

In OpenFOAM syntax::

    type        turbulentLengthScaleInlet14;
    value       uniform 0;

Coefficients:
    - Standard turbulent length scale inlet parameters (from base and earlier versions).
    - ``thermal_damp_coeff`` (float): Thermal damping for length scale. (default 0.02).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentLengthScaleInlet14BC"]


@BoundaryCondition.register("turbulentLengthScaleInlet14")
class TurbulentLengthScaleInlet14BC(BoundaryCondition):
    """Enhanced turbulent length scale inlet v14.

    - ``thermal_damp_coeff`` (float): Thermal damping for length scale. (default 0.02).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._wake_coeff = float(self._coeffs.get("wake_coeff", 0.1))
        self._pressure_grad_coeff = float(self._coeffs.get("pressure_grad_coeff", 0.05))
        self._curvature_coeff = float(self._coeffs.get("curvature_coeff", 0.03))
        self._thermal_damp_coeff = float(self._coeffs.get("thermal_damp_coeff", 0.02))

    @property
    def wake_coeff(self) -> float:
        return self._wake_coeff

    @property
    def pressure_grad_coeff(self) -> float:
        return self._pressure_grad_coeff

    @property
    def curvature_coeff(self) -> float:
        return self._curvature_coeff

    @property
    def thermal_damp_coeff(self) -> float:
        return self._thermal_damp_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced turbulent length scale inlet v14."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v14 enhancement: thermal damping for length scale.
        values = values * (1.0 - self._thermal_damp_coeff * 0.01)

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
        """Penalty method for v14 enhanced turbulent length scale inlet BC."""
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
