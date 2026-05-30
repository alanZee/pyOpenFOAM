"""Enhanced wave transmissive boundary condition (v12).

In OpenFOAM syntax::

    type        pressureWaveTransmissive12;
    value       uniform 0;

Coefficients:
    - Standard wave transmissive parameters (from base and earlier versions).
    - ``acoustic_scatter_coeff`` (float): Acoustic scattering correction. (default 0.02).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureWaveTransmissive12BC"]


@BoundaryCondition.register("pressureWaveTransmissive12")
class PressureWaveTransmissive12BC(BoundaryCondition):
    """Enhanced wave transmissive v12.

    - ``acoustic_scatter_coeff`` (float): Acoustic scattering correction. (default 0.02).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._vorticity_damp_coeff = float(self._coeffs.get("vorticity_damp_coeff", 0.05))
        self._acoustic_scatter_coeff = float(self._coeffs.get("acoustic_scatter_coeff", 0.02))

    @property
    def vorticity_damp_coeff(self) -> float:
        return self._vorticity_damp_coeff

    @property
    def acoustic_scatter_coeff(self) -> float:
        return self._acoustic_scatter_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced wave transmissive v12."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v12 enhancement: acoustic scattering correction.
        Ma_local = values.abs().mean() / (343.0 + 1e-30)
        values = values * (1.0 + self._acoustic_scatter_coeff * Ma_local ** 2)

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
        """Penalty method for v12 enhanced wave transmissive BC."""
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
