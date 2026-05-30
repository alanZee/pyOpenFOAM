"""Enhanced scaled heat flux boundary condition (v15).

In OpenFOAM syntax::

    type        scaledHeatFlux15;
    value       uniform 0;

Coefficients:
    - Standard scaled heat flux parameters (from base and earlier versions).
    - ``film_coeff`` (float): Film cooling correction. (default 0.0).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ScaledHeatFlux15BC"]


@BoundaryCondition.register("scaledHeatFlux15")
class ScaledHeatFlux15BC(BoundaryCondition):
    """Enhanced scaled heat flux v15.

    - ``film_coeff`` (float): Film cooling correction. (default 0.0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._history_coeff = float(self._coeffs.get("history_coeff", 0.1))
        self._spatial_period_coeff = float(self._coeffs.get("spatial_period_coeff", 0.0))
        self._contact_coeff = float(self._coeffs.get("contact_coeff", 0.0))
        self._beta_eps = float(self._coeffs.get("beta_eps", 0.0))
        self._turb_prandtl_coeff = float(self._coeffs.get("turb_prandtl_coeff", 0.85))
        self._wall_funct_coeff = float(self._coeffs.get("wall_funct_coeff", 0.0))
        self._volumetric_coeff = float(self._coeffs.get("volumetric_coeff", 0.0))
        self._film_coeff = float(self._coeffs.get("film_coeff", 0.0))

    @property
    def history_coeff(self) -> float:
        return self._history_coeff

    @property
    def spatial_period_coeff(self) -> float:
        return self._spatial_period_coeff

    @property
    def contact_coeff(self) -> float:
        return self._contact_coeff

    @property
    def beta_eps(self) -> float:
        return self._beta_eps

    @property
    def turb_prandtl_coeff(self) -> float:
        return self._turb_prandtl_coeff

    @property
    def wall_funct_coeff(self) -> float:
        return self._wall_funct_coeff

    @property
    def volumetric_coeff(self) -> float:
        return self._volumetric_coeff

    @property
    def film_coeff(self) -> float:
        return self._film_coeff
    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Apply enhanced scaled heat flux v15."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        owners = self._patch.owner_cells.to(device=device)
        values = field[owners]

        # v15 enhancement: film cooling correction.
        T_ref = 300.0
        values = values + self._film_coeff * (T_ref - values)

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
        """Penalty method for v15 enhanced scaled heat flux BC."""
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
