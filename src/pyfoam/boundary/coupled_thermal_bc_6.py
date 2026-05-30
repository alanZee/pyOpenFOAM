"""Enhanced coupled temperature boundary condition (v6).

In OpenFOAM syntax::

    type        coupledTemperature6;
    value       uniform 0;

Coefficients:
    - Standard coupled temperature parameters (from base and earlier versions).
    - ``convective_blend_coeff`` (float): Convective blending at interface. (default 1.0).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["CoupledTemperature6BC"]


@BoundaryCondition.register("coupledTemperature6")
class CoupledTemperature6BC(BoundaryCondition):
    """Enhanced coupled temperature v6.

    - ``convective_blend_coeff`` (float): Convective blending at interface. (default 1.0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._contact_resistance_coeff = float(self._coeffs.get("contact_resistance_coeff", 0.0))
        self._thermal_conductivity_ratio = float(self._coeffs.get("thermal_conductivity_ratio", 1.0))
        self._interfacial_htc = float(self._coeffs.get("interfacial_htc", 0.0))
        self._radiation_coeff = float(self._coeffs.get("radiation_coeff", 0.0))
        self._convective_blend_coeff = float(self._coeffs.get("convective_blend_coeff", 1.0))

    @property
    def contact_resistance_coeff(self) -> float:
        return self._contact_resistance_coeff

    @property
    def thermal_conductivity_ratio(self) -> float:
        return self._thermal_conductivity_ratio

    @property
    def interfacial_htc(self) -> float:
        return self._interfacial_htc

    @property
    def radiation_coeff(self) -> float:
        return self._radiation_coeff

    @property
    def convective_blend_coeff(self) -> float:
        return self._convective_blend_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced coupled temperature v6."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v6 enhancement: convective blending at interface.
        values = self._convective_blend_coeff * values + (1.0 - self._convective_blend_coeff) * values.mean()

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
        """Penalty method for v6 enhanced coupled temperature BC."""
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
