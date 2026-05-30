"""Enhanced wave transmissive boundary condition (v15).

In OpenFOAM syntax::

    type        pressureWaveTransmissive15;
    value       uniform 0;

Coefficients:
    - Standard wave transmissive parameters (from base and earlier versions).
    - ``relax_length_coeff`` (float): Adaptive relaxation length. (default 0.5).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureWaveTransmissive15BC"]


@BoundaryCondition.register("pressureWaveTransmissive15")
class PressureWaveTransmissive15BC(BoundaryCondition):
    """Enhanced wave transmissive v15.

    - ``relax_length_coeff`` (float): Adaptive relaxation length. (default 0.5).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._vorticity_damp_coeff = float(self._coeffs.get("vorticity_damp_coeff", 0.05))
        self._acoustic_scatter_coeff = float(self._coeffs.get("acoustic_scatter_coeff", 0.02))
        self._entropy_wave_coeff = float(self._coeffs.get("entropy_wave_coeff", 0.01))
        self._turb_damp_coeff = float(self._coeffs.get("turb_damp_coeff", 0.03))
        self._relax_length_coeff = float(self._coeffs.get("relax_length_coeff", 0.5))

    @property
    def vorticity_damp_coeff(self) -> float:
        return self._vorticity_damp_coeff

    @property
    def acoustic_scatter_coeff(self) -> float:
        return self._acoustic_scatter_coeff

    @property
    def entropy_wave_coeff(self) -> float:
        return self._entropy_wave_coeff

    @property
    def turb_damp_coeff(self) -> float:
        return self._turb_damp_coeff

    @property
    def relax_length_coeff(self) -> float:
        return self._relax_length_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced wave transmissive v15."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v15 enhancement: adaptive relaxation length.
        field_inf = 101325.0
        dp = (values - field_inf).abs()
        l_adaptive = 1.0 * (1.0 + self._relax_length_coeff * dp / (values.abs() + 1e-30))
        values = values * (1.0 - 0.01 * l_adaptive)

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
        """Penalty method for v15 enhanced wave transmissive BC."""
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
