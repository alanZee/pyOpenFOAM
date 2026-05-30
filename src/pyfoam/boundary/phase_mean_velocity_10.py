"""Enhanced phase mean velocity boundary condition (v10).

In OpenFOAM syntax::

    type        phaseMeanVelocity10;
    value       uniform 0;

Coefficients:
    - Standard phase mean velocity parameters (from base and earlier versions).
    - ``relax_coeff`` (float): Under-relaxation for stability. (default 1.0).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PhaseMeanVelocity10BC"]


@BoundaryCondition.register("phaseMeanVelocity10")
class PhaseMeanVelocity10BC(BoundaryCondition):
    """Enhanced phase mean velocity v10.

    - ``relax_coeff`` (float): Under-relaxation for stability. (default 1.0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._alpha_correction_coeff = float(self._coeffs.get("alpha_correction_coeff", 0.01))
        self._turb_mix_coeff = float(self._coeffs.get("turb_mix_coeff", 0.05))
        self._pressure_coupling_coeff = float(self._coeffs.get("pressure_coupling_coeff", 0.02))
        self._slip_velocity_coeff = float(self._coeffs.get("slip_velocity_coeff", 0.1))
        self._wall_correction_coeff = float(self._coeffs.get("wall_correction_coeff", 0.0))
        self._drag_correction_coeff = float(self._coeffs.get("drag_correction_coeff", 0.05))
        self._virtual_mass_coeff = float(self._coeffs.get("virtual_mass_coeff", 0.0))
        self._lift_coeff = float(self._coeffs.get("lift_coeff", 0.0))
        self._relax_coeff = float(self._coeffs.get("relax_coeff", 1.0))

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

    @property
    def wall_correction_coeff(self) -> float:
        return self._wall_correction_coeff

    @property
    def drag_correction_coeff(self) -> float:
        return self._drag_correction_coeff

    @property
    def virtual_mass_coeff(self) -> float:
        return self._virtual_mass_coeff

    @property
    def lift_coeff(self) -> float:
        return self._lift_coeff

    @property
    def relax_coeff(self) -> float:
        return self._relax_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced phase mean velocity v10."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v10 enhancement: under-relaxation for stability.
        values = self._relax_coeff * values + (1.0 - self._relax_coeff) * field[owners]

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
        """Penalty method for v10 enhanced phase mean velocity BC."""
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
