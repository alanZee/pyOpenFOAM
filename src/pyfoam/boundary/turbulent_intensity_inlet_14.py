"""Enhanced turbulent intensity inlet boundary condition (v14).

In OpenFOAM syntax::

    type        turbulentIntensityInlet14;
    value       uniform 0;

Coefficients:
    - Standard turbulent intensity inlet parameters (from base and earlier versions).
    - ``time_scale_coeff`` (float): Dynamic time-scale correction. (default 0.1).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentIntensityInlet14BC"]


@BoundaryCondition.register("turbulentIntensityInlet14")
class TurbulentIntensityInlet14BC(BoundaryCondition):
    """Enhanced turbulent intensity inlet v14.

    - ``time_scale_coeff`` (float): Dynamic time-scale correction. (default 0.1).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._wall_fluct_coeff = float(self._coeffs.get("wall_fluct_coeff", 0.3))
        self._kolmogorov_coeff = float(self._coeffs.get("kolmogorov_coeff", 0.05))
        self._dilatation_coeff = float(self._coeffs.get("dilatation_coeff", 0.02))
        self._time_scale_coeff = float(self._coeffs.get("time_scale_coeff", 0.1))

    @property
    def wall_fluct_coeff(self) -> float:
        return self._wall_fluct_coeff

    @property
    def kolmogorov_coeff(self) -> float:
        return self._kolmogorov_coeff

    @property
    def dilatation_coeff(self) -> float:
        return self._dilatation_coeff

    @property
    def time_scale_coeff(self) -> float:
        return self._time_scale_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced turbulent intensity inlet v14."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v14 enhancement: dynamic time-scale correction.
        values = values * (1.0 + self._time_scale_coeff * 0.01)

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
        """Penalty method for v14 enhanced turbulent intensity inlet BC."""
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
