"""Enhanced processor cyclic boundary condition (v9).

In OpenFOAM syntax::

    type        processorCyclic9;
    value       uniform 0;

Coefficients:
    - Standard processor cyclic parameters (from base and earlier versions).
    - ``anisotropy_coeff`` (float): Anisotropic coupling correction. (default 0.03).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ProcessorCyclic9BC"]


@BoundaryCondition.register("processorCyclic9")
class ProcessorCyclic9BC(BoundaryCondition):
    """Enhanced processor cyclic v9.

    - ``anisotropy_coeff`` (float): Anisotropic coupling correction. (default 0.03).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._buffer_blend_coeff = float(self._coeffs.get("buffer_blend_coeff", 0.1))
        self._transform_correct_coeff = float(self._coeffs.get("transform_correct_coeff", 0.05))
        self._face_reorder_coeff = float(self._coeffs.get("face_reorder_coeff", 0.0))
        self._ghost_cell_coeff = float(self._coeffs.get("ghost_cell_coeff", 0.1))
        self._conservation_coeff = float(self._coeffs.get("conservation_coeff", 0.01))
        self._smoothness_coeff = float(self._coeffs.get("smoothness_coeff", 0.05))
        self._gradient_correct_coeff = float(self._coeffs.get("gradient_correct_coeff", 0.02))
        self._anisotropy_coeff = float(self._coeffs.get("anisotropy_coeff", 0.03))

    @property
    def buffer_blend_coeff(self) -> float:
        return self._buffer_blend_coeff

    @property
    def transform_correct_coeff(self) -> float:
        return self._transform_correct_coeff

    @property
    def face_reorder_coeff(self) -> float:
        return self._face_reorder_coeff

    @property
    def ghost_cell_coeff(self) -> float:
        return self._ghost_cell_coeff

    @property
    def conservation_coeff(self) -> float:
        return self._conservation_coeff

    @property
    def smoothness_coeff(self) -> float:
        return self._smoothness_coeff

    @property
    def gradient_correct_coeff(self) -> float:
        return self._gradient_correct_coeff

    @property
    def anisotropy_coeff(self) -> float:
        return self._anisotropy_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced processor cyclic v9."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v9 enhancement: anisotropic coupling correction.
        idx = torch.linspace(-0.5, 0.5, n, device=device, dtype=dtype)
        values = values * (1.0 + self._anisotropy_coeff * idx)

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
        """Penalty method for v9 enhanced processor cyclic BC."""
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
