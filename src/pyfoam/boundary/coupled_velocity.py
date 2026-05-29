"""
Coupled velocity boundary condition for conjugate heat transfer (CHT).

This boundary condition enforces velocity coupling at the interface between
two regions (e.g., fluid and solid) in conjugate heat transfer simulations.
The velocity at the boundary face is set to match the velocity of the coupled
face in the adjacent region.

For conjugate heat transfer:
- Velocity continuity: U_fluid = U_solid at the interface
- Traction continuity: sigma·n_fluid = -sigma·n_solid at the interface

In OpenFOAM syntax::

    type        coupledVelocity;
    neighbourRegion  solid;
    neighbourPatch   fluid_to_solid;
    value       uniform (0 0 0);

Usage::

    bc = BoundaryCondition.create("coupledVelocity", patch, coeffs={
        "neighbourRegion": "solid",
        "neighbourPatch": "fluid_to_solid",
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["CoupledVelocityBC"]


@BoundaryCondition.register("coupledVelocity")
class CoupledVelocityBC(BoundaryCondition):
    """Coupled velocity boundary condition for CHT interfaces.

    Enforces velocity continuity by reading the velocity from the coupled
    face in the adjacent region.  The coupled region's field values are
    mapped to this patch via face-to-face correspondence.

    This BC is designed for use at fluid-solid interfaces in conjugate
heat transfer simulations where velocity must be coupled between regions.

    Coefficients:
        - ``neighbourRegion``: Name of the coupled region (informational).
        - ``neighbourPatch``: Name of the coupled patch (informational).
        - ``value``: Initial velocity (used for shape, overwritten on apply).
        - ``coupledField``: Optional tensor of coupled velocity values
          ``(n_faces, 3)`` injected at runtime.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._neighbour_region = self._coeffs.get("neighbourRegion", "")
        self._neighbour_patch = self._coeffs.get("neighbourPatch", "")
        # Optional coupled field injected at runtime
        self._coupled_field: torch.Tensor | None = self._coeffs.get("coupledField")

    @property
    def neighbour_region(self) -> str:
        """Name of the coupled region."""
        return self._neighbour_region

    @property
    def neighbour_patch(self) -> str:
        """Name of the coupled patch."""
        return self._neighbour_patch

    @property
    def coupled_field(self) -> torch.Tensor | None:
        """Coupled velocity field ``(n_faces, 3)`` or ``None``."""
        return self._coupled_field

    @coupled_field.setter
    def coupled_field(self, value: torch.Tensor | None) -> None:
        """Set the coupled velocity field."""
        self._coupled_field = value

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocity from the coupled region.

        If a coupled field has been injected, uses those values directly.
        Otherwise, falls back to zero-gradient (copies owner cell values).

        Args:
            field: Velocity field ``(n_total, 3)``.
            patch_idx: Optional start index into *field*.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if self._coupled_field is not None:
            # Use coupled field values
            coupled_vals = self._coupled_field[:n].to(device=device, dtype=dtype)
            if patch_idx is not None:
                field[patch_idx : patch_idx + n] = coupled_vals
            else:
                field[self._patch.face_indices] = coupled_vals
        else:
            # Fallback: zero-gradient (copy owner cell values)
            owners = self._patch.owner_cells.to(device=device)
            owner_vals = field[owners]
            if patch_idx is not None:
                field[patch_idx : patch_idx + n] = owner_vals
            else:
                field[self._patch.face_indices] = owner_vals

        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for coupled velocity BC.

        When coupled field is available: large diagonal penalty + source
        matching coupled values.  Otherwise: zero contribution (zero-gradient
        is handled by the velocity equation).
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        if self._coupled_field is None:
            return diag, source

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas

        # Penalty diagonal
        diag.scatter_add_(0, owners, coeff)

        # Source: project coupled velocity onto x-component (scalar matrix)
        coupled_vals = self._coupled_field[:self._patch.n_faces].to(
            device=device, dtype=dtype,
        )
        source.scatter_add_(0, owners, coeff * coupled_vals[:, 0])

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
