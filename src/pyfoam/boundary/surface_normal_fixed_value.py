"""
surfaceNormalFixedValue boundary condition.

Velocity inlet that specifies only the normal component; the
tangential components are zero.  In OpenFOAM syntax::

    type                  surfaceNormalFixedValue;
    value                 uniform (1 0 0);   # magnitude in x (unused)

The actual value applied is::

    v_face = magnitude * face_normal

where *magnitude* is taken from ``coeffs["value"]`` (scalar) or
from a per-face tensor.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition

__all__ = ["SurfaceNormalFixedValueBC"]


@BoundaryCondition.register("surfaceNormalFixedValue")
class SurfaceNormalFixedValueBC(BoundaryCondition):
    """Surface-normal fixed-value velocity inlet.

    Prescribes a velocity that is purely in the outward-normal
    direction of each boundary face.  The magnitude is uniform
    across the patch.
    """

    def __init__(self, patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._magnitude = self._resolve_magnitude()

    def _resolve_magnitude(self) -> torch.Tensor:
        """Parse the ``value`` coefficient into a scalar magnitude.

        Accepts a scalar number or a ``(n_faces,)`` tensor.
        Vector inputs are reduced to their norm.
        """
        raw = self._coeffs.get("value", 0.0)
        if isinstance(raw, torch.Tensor):
            t = raw.to(dtype=get_default_dtype(), device=get_device())
            if t.dim() == 0:
                # Scalar tensor
                return t
            if t.dim() == 1 and t.shape[0] == self._patch.n_faces:
                return t
            if t.dim() >= 2:
                # Vector per face → take norm
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

    def _face_velocity(self, device: torch.device | None = None) -> torch.Tensor:
        """Compute ``(n_faces, 3)`` velocity = magnitude * normal."""
        normals = self._patch.face_normals.to(
            device=device, dtype=get_default_dtype()
        )
        mag = self._magnitude.to(device=device, dtype=get_default_dtype())
        # Broadcast: scalar magnitude × (n, 3) normals
        return mag.unsqueeze(-1) * normals if mag.dim() == 0 else mag.unsqueeze(-1) * normals

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocity to magnitude * face_normal.

        Works for both scalar fields (stores magnitude) and
        vector fields ``(n, 3)`` (stores full velocity vector).
        """
        if field.dim() >= 2:
            # Vector field: set full velocity
            vel = self._face_velocity(device=field.device)
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = vel
            else:
                field[self._patch.face_indices] = vel
        else:
            # Scalar field: set magnitude
            mag = self._magnitude.to(device=field.device, dtype=field.dtype)
            if mag.dim() == 0:
                values = mag.expand(self._patch.n_faces)
            else:
                values = mag
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
        """Penalty method (same as fixedValue).

        For each boundary face adjacent to cell *c*:
            diag[c]   += deltaCoeff * faceArea
            source[c] += deltaCoeff * faceArea * magnitude
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
        mag = self._magnitude.to(device=device, dtype=dtype)
        if mag.dim() == 0:
            mag = mag.expand(self._patch.n_faces)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * mag)

        return diag, source
