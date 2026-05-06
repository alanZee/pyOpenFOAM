"""
fixedGradient boundary condition.

Prescribes the normal gradient at the boundary face.  In OpenFOAM syntax::

    type      fixedGradient;
    gradient  uniform 10;

The face value is extrapolated from the internal field plus the
prescribed gradient times the distance to the cell centre.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["FixedGradientBC"]


@BoundaryCondition.register("fixedGradient")
class FixedGradientBC(BoundaryCondition):
    """Fixed-gradient (Neumann) boundary condition.

    Prescribes ∂φ/∂n = gradient at each boundary face.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._gradient = self._resolve_gradient()

    def _resolve_gradient(self) -> torch.Tensor:
        """Parse the ``gradient`` coefficient into a tensor."""
        raw = self._coeffs.get("gradient", 0.0)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    @property
    def gradient(self) -> torch.Tensor:
        """Return the prescribed gradient values."""
        return self._gradient

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Extrapolate face values: φ_face = φ_cell + grad * d.

        where d = 1/deltaCoeff is the distance from cell centre to face.
        """
        owners = self._patch.owner_cells.to(device=field.device)
        deltas = self._patch.delta_coeffs.to(device=field.device, dtype=field.dtype)
        grad = self._gradient.to(device=field.device, dtype=field.dtype)

        # d = 1 / deltaCoeff
        dist = 1.0 / deltas
        owner_values = field[owners]
        face_values = owner_values + grad * dist

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = face_values
        else:
            field[self._patch.face_indices] = face_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flux contribution from prescribed gradient.

        For fixed gradient, the flux through the face is:

            flux = area * gradient

        This enters the source term (explicit contribution):

            source[c] += area * gradient

        No implicit (diagonal) contribution.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        grad = self._gradient.to(device=device, dtype=dtype)

        source.scatter_add_(0, owners, areas * grad)

        return diag, source
