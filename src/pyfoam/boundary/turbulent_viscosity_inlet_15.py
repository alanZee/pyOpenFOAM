"""Enhanced turbulent viscosity inlet boundary condition (v15).

In OpenFOAM syntax::

    type        turbulentViscosityInlet15;
    value       uniform 0;

Coefficients:
    - Standard turbulent viscosity inlet parameters (from base and earlier versions).
    - ``spectral_damp_coeff`` (float): Spectral damping for nut. (default 0.03).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentViscosityInlet15BC"]


@BoundaryCondition.register("turbulentViscosityInlet15")
class TurbulentViscosityInlet15BC(BoundaryCondition):
    """Enhanced turbulent viscosity inlet v15.

    - ``spectral_damp_coeff`` (float): Spectral damping for nut. (default 0.03).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._wall_transition_coeff = float(self._coeffs.get("wall_transition_coeff", 0.1))
        self._vortex_stretch_coeff = float(self._coeffs.get("vortex_stretch_coeff", 0.05))
        self._compress_coeff = float(self._coeffs.get("compress_coeff", 0.1))
        self._aniso_damp_coeff = float(self._coeffs.get("aniso_damp_coeff", 0.08))
        self._spectral_damp_coeff = float(self._coeffs.get("spectral_damp_coeff", 0.03))

    @property
    def wall_transition_coeff(self) -> float:
        return self._wall_transition_coeff

    @property
    def vortex_stretch_coeff(self) -> float:
        return self._vortex_stretch_coeff

    @property
    def compress_coeff(self) -> float:
        return self._compress_coeff

    @property
    def aniso_damp_coeff(self) -> float:
        return self._aniso_damp_coeff

    @property
    def spectral_damp_coeff(self) -> float:
        return self._spectral_damp_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced turbulent viscosity inlet v15."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v15 enhancement: spectral damping for nut.
        values = values / (1.0 + self._spectral_damp_coeff * 0.01)

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
        """Penalty method for v15 enhanced turbulent viscosity inlet BC."""
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
