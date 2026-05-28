"""
Symmetry/slip combined boundary condition.

Behaves like slip (free-slip wall) but additionally validates that the patch
lies on a geometric symmetry plane, analogous to the ``symmetryPlane`` BC.

In OpenFOAM syntax::

    type   symmetrySlip;

For scalar fields: equivalent to zeroGradient.
For vector fields: the normal component is zeroed and the tangential
component is preserved (free slip), with a check that all face normals
are parallel (i.e. the patch is a plane).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["SymmetrySlipBC"]

logger = logging.getLogger(__name__)


@BoundaryCondition.register("symmetrySlip")
class SymmetrySlipBC(BoundaryCondition):
    """Combined symmetry/slip boundary condition.

    Behaves like ``slip`` (free-slip wall) but additionally validates that
    the patch is a geometric symmetry plane (all face normals are parallel).

    The apply() method:
    - Scalar fields: zero-gradient (copy owner values).
    - Vector fields: remove the normal component, preserve tangential.

    Matrix contributions are zero (no friction / free-slip).
    """

    def __init__(
        self,
        patch: Patch,
        coeffs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(patch, coeffs)
        self._validate_symmetry_plane()

    def _validate_symmetry_plane(self) -> None:
        """Validate that the patch lies on a geometric symmetry plane.

        Checks that all face normals are parallel (or anti-parallel)
        to the mean normal direction.  Issues a warning if not.
        """
        normals = self._patch.face_normals
        if normals.shape[0] < 2:
            return

        mean_normal = normals.mean(dim=0)
        mean_norm = mean_normal.norm()

        if mean_norm < 1e-30:
            logger.warning(
                "SymmetrySlipBC: patch '%s' has near-zero mean normal",
                self._patch.name,
            )
            return

        mean_normal = mean_normal / mean_norm

        for i in range(normals.shape[0]):
            n_i = normals[i]
            n_i_norm = n_i.norm()
            if n_i_norm < 1e-30:
                continue
            cos_angle = torch.dot(n_i / n_i_norm, mean_normal).abs()
            if cos_angle < 0.99:
                logger.warning(
                    "SymmetrySlipBC: patch '%s' face %d is not parallel "
                    "to mean normal (cos=%.4f).  This may not be a "
                    "geometric symmetry plane.",
                    self._patch.name, i, cos_angle.item(),
                )
                break

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Remove normal component, preserve tangential (free slip).

        For scalar fields (1-D): zero-gradient (copy owner values).
        For vector fields (2-D per component): removes the normal component.

            phi_face = phi_cell - (phi_cell . n) n
        """
        owners = self._patch.owner_cells.to(device=field.device)
        normals = self._patch.face_normals.to(device=field.device, dtype=field.dtype)

        if field.dim() == 1:
            owner_values = field[owners]
            if patch_idx is not None:
                n = self._patch.n_faces
                field[patch_idx : patch_idx + n] = owner_values
            else:
                field[self._patch.face_indices] = owner_values
        else:
            owner_values = field[owners]
            normal_comp = (owner_values * normals).sum(dim=-1, keepdim=True)
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
        """Free-slip: zero matrix contributions (no flux coupling)."""
        device = get_device()
        dtype = get_default_dtype()
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source
