"""
fixedNormal boundary condition.

Fixes only the normal component of a vector field at the boundary while
preserving the tangential components.  In OpenFOAM syntax::

    type    fixedNormal;
    value   uniform 1;     # prescribed normal component magnitude

For scalar fields, equivalent to fixedValue.
For vector fields, the normal component is set to the prescribed value
and tangential components are copied from the adjacent owner cell.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["FixedNormalBC"]


@BoundaryCondition.register("fixedNormal")
class FixedNormalBC(BoundaryCondition):
    """Fixed-normal boundary condition.

    Sets only the normal component of a vector field to a prescribed value,
    preserving the tangential components from the adjacent owner cell.

    For scalar fields, behaves like fixedValue.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._value = self._resolve_value()

    def _resolve_value(self) -> torch.Tensor:
        """Parse the ``value`` coefficient into a tensor."""
        raw = self._coeffs.get("value", 0.0)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.tensor(float(raw), dtype=get_default_dtype(), device=get_device())

    @property
    def value(self) -> torch.Tensor:
        """Return the prescribed normal-component value."""
        return self._value

    @value.setter
    def value(self, new_value: float | torch.Tensor) -> None:
        """Update the prescribed normal-component value."""
        if isinstance(new_value, torch.Tensor):
            self._value = new_value.to(dtype=get_default_dtype(), device=get_device())
        else:
            self._value = torch.tensor(
                float(new_value), dtype=get_default_dtype(), device=get_device()
            )

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Fix the normal component, preserve tangential.

        For scalar fields: sets boundary values to the prescribed value.
        For vector fields: replaces the normal component with the prescribed
        value while keeping tangential components from the owner cell.
        """
        owners = self._patch.owner_cells.to(device=field.device)
        normals = self._patch.face_normals.to(device=field.device, dtype=field.dtype)

        if field.dim() >= 2:
            # Vector field: owner values, replace normal component
            owner_values = field[owners]  # (n_faces, 3)
            normal_comp = (owner_values * normals).sum(dim=-1, keepdim=True)
            tangential = owner_values - normal_comp * normals

            val = self._value.to(device=field.device, dtype=field.dtype)
            # val is scalar → broadcast to (n_faces, 1) for normal direction
            if val.dim() == 0:
                new_normal = val * normals
            else:
                new_normal = val.unsqueeze(-1) * normals

            result = tangential + new_normal

            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = result
            else:
                field[self._patch.face_indices] = result
        else:
            # Scalar field: fixed value
            val = self._value.to(device=field.device, dtype=field.dtype)
            if val.dim() == 0:
                values = val.expand(self._patch.n_faces)
            else:
                values = val
            if patch_idx is not None:
                n = self._patch.n_faces
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
        """Penalty method for the normal component only.

        For vector fields, contributes only for the normal direction.
        For scalar fields, standard penalty (same as fixedValue).
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        val = self._value.to(device=device, dtype=dtype)

        # Implicit coefficient = deltaCoeff * area
        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        if val.dim() == 0:
            source.scatter_add_(0, owners, coeff * val)
        else:
            source.scatter_add_(0, owners, coeff * val)

        return diag, source
