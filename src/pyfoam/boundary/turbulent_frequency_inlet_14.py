"""Enhanced turbulent frequency inlet boundary condition (v14).

In OpenFOAM syntax::

    type        turbulentFrequencyInlet14;
    value       uniform 0;

Coefficients:
    - Standard turbulent frequency inlet parameters (from base and earlier versions).
    - ``cross_diff_coeff`` (float): Cross-diffusion correction for omega. (default 0.03).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentFrequencyInlet14BC"]


@BoundaryCondition.register("turbulentFrequencyInlet14")
class TurbulentFrequencyInlet14BC(BoundaryCondition):
    """Enhanced turbulent frequency inlet v14.

    - ``cross_diff_coeff`` (float): Cross-diffusion correction for omega. (default 0.03).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._cross_diff_coeff = float(self._coeffs.get("cross_diff_coeff", 0.03))

    @property
    def cross_diff_coeff(self) -> float:
        return self._cross_diff_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced turbulent frequency inlet v14."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v14 enhancement: cross-diffusion correction for omega.
        values = values + self._cross_diff_coeff * 0.01

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
        """Penalty method for v14 enhanced turbulent frequency inlet BC."""
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
