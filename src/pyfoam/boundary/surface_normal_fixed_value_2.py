"""
Enhanced surface normal fixed value boundary condition — version 2.

An improved version of ``surfaceNormalFixedValue`` that adds:

1. **Per-face magnitude variation**: Supports spatially varying
   magnitude via a scaling function.

2. **Tangential component**: Optionally preserves a fraction of the
   tangential velocity from the interior, allowing partial slip.

In OpenFOAM syntax::

    type              surfaceNormalFixedValue2;
    value             uniform 1.0;           // normal velocity magnitude
    tangentialFraction  0.0;                 // fraction of tangential kept
    value               uniform (1 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["SurfaceNormalFixedValue2BC"]


@BoundaryCondition.register("surfaceNormalFixedValue2")
class SurfaceNormalFixedValue2BC(BoundaryCondition):
    """Enhanced surface-normal fixed-value velocity BC (version 2).

    Prescribes a velocity in the outward-normal direction with optional
    partial preservation of the tangential velocity component from the
    interior.  This allows ``surfaceNormalFixedValue2`` to act as a
    partial-slip normal inlet.

    Coefficients:
        - ``value``: Normal velocity magnitude (scalar or per-face).
        - ``tangentialFraction``: Fraction (0-1) of interior tangential
          velocity to keep (default 0.0 = pure normal, 1.0 = full slip).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._magnitude = self._resolve_magnitude()
        self._tangential_fraction = float(
            self._coeffs.get("tangentialFraction", 0.0)
        )

    def _resolve_magnitude(self) -> torch.Tensor:
        """Parse the ``value`` coefficient into a scalar magnitude."""
        raw = self._coeffs.get("value", 0.0)
        if isinstance(raw, torch.Tensor):
            t = raw.to(dtype=get_default_dtype(), device=get_device())
            if t.dim() == 0:
                return t
            if t.dim() == 1 and t.shape[0] == self._patch.n_faces:
                return t
            if t.dim() >= 2:
                return t.norm(dim=-1)
            return t
        return torch.tensor(float(raw), dtype=get_default_dtype(), device=get_device())

    @property
    def magnitude(self) -> torch.Tensor:
        """Return the normal-velocity magnitude."""
        return self._magnitude

    @magnitude.setter
    def magnitude(self, new_mag: float | torch.Tensor) -> None:
        """Update the magnitude."""
        if isinstance(new_mag, torch.Tensor):
            self._magnitude = new_mag.to(dtype=get_default_dtype(), device=get_device())
        else:
            self._magnitude = torch.tensor(
                float(new_mag), dtype=get_default_dtype(), device=get_device()
            )

    @property
    def tangential_fraction(self) -> float:
        """Return tangential velocity preservation fraction."""
        return self._tangential_fraction

    def _face_velocity(
        self,
        device: torch.device | None = None,
        interior_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``(n_faces, 3)`` velocity with optional tangential component.

        v = mag * n + tangentialFraction * (U_interior - (U_interior . n) * n)
        """
        normals = self._patch.face_normals.to(
            device=device, dtype=get_default_dtype()
        )
        mag = self._magnitude.to(device=device, dtype=get_default_dtype())

        # Normal component: magnitude * normal
        normal_vel = mag.unsqueeze(-1) * normals if mag.dim() > 0 else mag * normals

        # Tangential component from interior
        if (
            interior_velocity is not None
            and self._tangential_fraction > 0.0
        ):
            u_int = interior_velocity.to(device=device, dtype=get_default_dtype())
            # Project interior velocity onto normal
            u_normal = (u_int * normals).sum(dim=-1, keepdim=True) * normals
            # Tangential = interior - normal projection
            u_tangent = u_int - u_normal
            return normal_vel + self._tangential_fraction * u_tangent

        return normal_vel

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        interior_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face velocity with optional tangential preservation."""
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if field.dim() >= 2:
            owners = self._patch.owner_cells.to(device=device)
            int_vel = interior_velocity if interior_velocity is not None else field[owners]
            vel = self._face_velocity(device=device, interior_velocity=int_vel).to(dtype=dtype)
            if patch_idx is not None:
                field[patch_idx : patch_idx + n] = vel
            else:
                field[self._patch.face_indices] = vel
        else:
            mag = self._magnitude.to(device=device, dtype=dtype)
            if mag.dim() == 0:
                values = mag.expand(n)
            else:
                values = mag
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
        """Penalty method (same as surfaceNormalFixedValue)."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        mag = self._magnitude.to(device=device, dtype=dtype)
        if mag.dim() == 0:
            mag = mag.expand(self._patch.n_faces)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * mag)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
