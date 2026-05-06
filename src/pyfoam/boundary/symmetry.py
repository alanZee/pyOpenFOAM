"""
symmetryPlane boundary condition.

Enforces that the normal component of a vector (or gradient of a scalar)
is zero at a symmetry plane.  In OpenFOAM syntax::

    type   symmetryPlane;

For scalar fields this is equivalent to zeroGradient.
For vector fields, the normal component is zeroed and the tangential
component is unconstrained (like slip).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["SymmetryBC"]


@BoundaryCondition.register("symmetryPlane")
class SymmetryBC(BoundaryCondition):
    """Symmetry-plane boundary condition.

    Removes the normal component of a vector field at the boundary
    while leaving the tangential component unchanged.
    """

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Project the field onto the tangent plane.

        For scalar fields (1-D): delegates to zeroGradient behaviour.
        For vector fields (2-D per component): removes the normal component.

            φ_face = φ_cell - (φ_cell · n) n
        """
        owners = self._patch.owner_cells.to(device=field.device)
        normals = self._patch.face_normals.to(device=field.device, dtype=field.dtype)

        if field.dim() == 1:
            # Scalar: zero-gradient (copy owner values)
            owner_values = field[owners]
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = owner_values
            else:
                field[self._patch.face_indices] = owner_values
        else:
            # Vector field: shape (n_faces, 3)
            owner_values = field[owners]  # (n_faces, 3)
            # Normal component: (φ · n) per face
            normal_comp = (owner_values * normals).sum(dim=-1, keepdim=True)
            # Remove normal component
            projected = owner_values - normal_comp * normals

            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = projected
            else:
                field[self._patch.face_indices] = projected

        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetry plane: no matrix contribution (zero flux)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source
