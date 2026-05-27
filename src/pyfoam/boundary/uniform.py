"""
uniform boundary condition.

Broadcasts a single scalar value to all boundary faces.
In OpenFOAM syntax::

    type    uniform;
    value   uniform 1;

The matrix contribution uses the penalty method: a large diagonal
coefficient (``deltaCoeff * faceArea``) and a matching source term.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["UniformBC"]


@BoundaryCondition.register("uniform")
class UniformBC(BoundaryCondition):
    """Uniform boundary condition.

    Prescribes a single uniform value broadcast to all boundary faces.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._value = self._resolve_value()

    def _resolve_value(self) -> float:
        """Parse the ``value`` coefficient as a scalar."""
        raw = self._coeffs.get("value", 0.0)
        if isinstance(raw, torch.Tensor):
            return float(raw.item())
        return float(raw)

    @property
    def value(self) -> float:
        """Return the uniform scalar value."""
        return self._value

    @value.setter
    def value(self, new_value: float) -> None:
        """Update the uniform value."""
        self._value = float(new_value)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Broadcast scalar to all boundary faces."""
        values = torch.full(
            (self._patch.n_faces,),
            self._value,
            dtype=field.dtype,
            device=field.device,
        )
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
        """Penalty method: large diagonal + matching source."""
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
        source.scatter_add_(0, owners, coeff * self._value)

        return diag, source
