"""Enhanced phase mean velocity boundary condition (v5).

In OpenFOAM syntax::

    type        phaseMeanVelocity5;
    value       uniform 0;

Coefficients:
    - Standard phase mean velocity parameters (from base and earlier versions).
    - ``slip_velocity_coeff`` (float): Slip velocity correction. (default 0.1).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PhaseMeanVelocity5BC"]


@BoundaryCondition.register("phaseMeanVelocity5")
class PhaseMeanVelocity5BC(BoundaryCondition):
    """Enhanced phase mean velocity v5.

    - ``slip_velocity_coeff`` (float): Slip velocity correction. (default 0.1).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._alpha_correction_coeff = float(self._coeffs.get("alpha_correction_coeff", 0.01))
        self._turb_mix_coeff = float(self._coeffs.get("turb_mix_coeff", 0.05))
        self._pressure_coupling_coeff = float(self._coeffs.get("pressure_coupling_coeff", 0.02))
        self._slip_velocity_coeff = float(self._coeffs.get("slip_velocity_coeff", 0.1))

    @property
    def alpha_correction_coeff(self) -> float:
        return self._alpha_correction_coeff

    @property
    def turb_mix_coeff(self) -> float:
        return self._turb_mix_coeff

    @property
    def pressure_coupling_coeff(self) -> float:
        return self._pressure_coupling_coeff

    @property
    def slip_velocity_coeff(self) -> float:
        return self._slip_velocity_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced phase mean velocity v5."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v5 enhancement: slip velocity correction.
        values = values * (1.0 + self._slip_velocity_coeff * 0.01)

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
        """Penalty method for v5 enhanced phase mean velocity BC."""
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
