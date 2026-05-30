"""Enhanced slip wall boundary condition (v9).

In OpenFOAM syntax::

    type        slip9;
    value       uniform 0;

Coefficients:
    - Standard slip wall parameters (from base and earlier versions).
    - ``compress_coeff`` (float): Compressibility correction at slip wall. (default 0.0).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["Slip9BC"]


@BoundaryCondition.register("slip9")
class Slip9BC(BoundaryCondition):
    """Enhanced slip wall v9.

    - ``compress_coeff`` (float): Compressibility correction at slip wall. (default 0.0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._tangential_correction = float(self._coeffs.get("tangential_correction", 0.0))
        self._normal_correction = float(self._coeffs.get("normal_correction", 0.0))
        self._viscous_sublayer_coeff = float(self._coeffs.get("viscous_sublayer_coeff", 0.0))
        self._roughness_coeff = float(self._coeffs.get("roughness_coeff", 0.0))
        self._pressure_gradient_coeff = float(self._coeffs.get("pressure_gradient_coeff", 0.0))
        self._turbulence_damp_coeff = float(self._coeffs.get("turbulence_damp_coeff", 0.0))
        self._curvature_coeff = float(self._coeffs.get("curvature_coeff", 0.0))
        self._compress_coeff = float(self._coeffs.get("compress_coeff", 0.0))

    @property
    def tangential_correction(self) -> float:
        return self._tangential_correction

    @property
    def normal_correction(self) -> float:
        return self._normal_correction

    @property
    def viscous_sublayer_coeff(self) -> float:
        return self._viscous_sublayer_coeff

    @property
    def roughness_coeff(self) -> float:
        return self._roughness_coeff

    @property
    def pressure_gradient_coeff(self) -> float:
        return self._pressure_gradient_coeff

    @property
    def turbulence_damp_coeff(self) -> float:
        return self._turbulence_damp_coeff

    @property
    def curvature_coeff(self) -> float:
        return self._curvature_coeff

    @property
    def compress_coeff(self) -> float:
        return self._compress_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced slip wall v9."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v9 enhancement: compressibility correction at slip wall.
        values = values * (1.0 + self._compress_coeff)

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
        """Penalty method for v9 enhanced slip wall BC."""
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
