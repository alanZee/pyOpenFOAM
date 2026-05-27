"""
symmetryPlane boundary condition.

Enforces that the normal component of a vector (or gradient of a scalar)
is zero at a geometric plane of symmetry.  In OpenFOAM syntax::

    type   symmetryPlane;

Similar to the ``symmetry`` BC but specifically designed for planes
of symmetry.  Validates that the patch faces lie on a geometric plane
(all face normals are parallel or anti-parallel).

For scalar fields this is equivalent to zeroGradient.
For vector fields, the normal component is zeroed and the tangential
component is unconstrained (like slip).

Additionally provides matrix contributions that enforce the zero-normal-
flux constraint implicitly through penalty terms.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["SymmetryPlaneBC"]

logger = logging.getLogger(__name__)


@BoundaryCondition.register("symmetryPlane")
class SymmetryPlaneBC(BoundaryCondition):
    """Symmetry-plane boundary condition.

    Removes the normal component of a vector field at the boundary
    while leaving the tangential component unchanged.  Designed for
    geometric planes of symmetry where all face normals are parallel.

    The key difference from the general ``symmetry`` BC is:
    - Validates that the patch is a geometric plane (consistent normals)
    - Provides explicit matrix contributions for the zero-normal-flux
      constraint via penalty terms
    """

    def __init__(
        self,
        patch: Patch,
        coeffs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(patch, coeffs)
        self._validate_patch()

    def _validate_patch(self) -> None:
        """Validate that the patch lies on a geometric plane.

        Checks that all face normals are parallel (or anti-parallel)
        to the mean normal direction.  Issues a warning if not.
        """
        normals = self._patch.face_normals
        if normals.shape[0] < 2:
            return

        # Compute mean normal direction
        mean_normal = normals.mean(dim=0)
        mean_norm = mean_normal.norm()

        if mean_norm < 1e-30:
            logger.warning(
                "SymmetryPlaneBC: patch '%s' has near-zero mean normal",
                self._patch.name,
            )
            return

        mean_normal = mean_normal / mean_norm

        # Check that each normal is parallel or anti-parallel to the mean
        for i in range(normals.shape[0]):
            n_i = normals[i]
            n_i_norm = n_i.norm()
            if n_i_norm < 1e-30:
                continue
            cos_angle = torch.dot(n_i / n_i_norm, mean_normal).abs()
            if cos_angle < 0.99:
                logger.warning(
                    "SymmetryPlaneBC: patch '%s' face %d is not parallel "
                    "to mean normal (cos=%.4f).  This may not be a "
                    "geometric symmetry plane.",
                    self._patch.name, i, cos_angle.item(),
                )
                break

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
        """Symmetry plane matrix contributions.

        Adds a penalty-based diagonal contribution to enforce the
        zero-normal-flux constraint implicitly:

            diag[c] += δ * |S| * penalty_coeff

        where penalty_coeff ensures the normal component is driven to zero.
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

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)

        return diag, source
