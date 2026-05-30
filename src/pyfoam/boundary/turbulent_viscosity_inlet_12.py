"""Enhanced turbulent viscosity inlet boundary condition (v12).

In OpenFOAM syntax::

    type        turbulentViscosityInlet12;
    value       uniform 0;

Coefficients:
    - Standard turbulent viscosity inlet parameters (from base and earlier versions).
    - ``vortex_stretch_coeff`` (float): Vortex stretching correction. (default 0.05).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentViscosityInlet12BC"]


@BoundaryCondition.register("turbulentViscosityInlet12")
class TurbulentViscosityInlet12BC(BoundaryCondition):
    """Enhanced turbulent viscosity inlet v12.

    - ``vortex_stretch_coeff`` (float): Vortex stretching correction. (default 0.05).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._wall_transition_coeff = float(self._coeffs.get("wall_transition_coeff", 0.1))
        self._vortex_stretch_coeff = float(self._coeffs.get("vortex_stretch_coeff", 0.05))

    @property
    def wall_transition_coeff(self) -> float:
        return self._wall_transition_coeff

    @property
    def vortex_stretch_coeff(self) -> float:
        return self._vortex_stretch_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced turbulent viscosity inlet v12."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v12 enhancement: vortex stretching correction.
        values = values * (1.0 + self._vortex_stretch_coeff * 0.01)

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
        """Penalty method for v12 enhanced turbulent viscosity inlet BC."""
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
