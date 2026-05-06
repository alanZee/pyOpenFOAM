"""
fixedValue boundary condition.

Sets boundary-face values to a prescribed value.  In OpenFOAM syntax::

    type    fixedValue;
    value   uniform 1;

The matrix contribution uses the penalty method: a large diagonal
coefficient (``deltaCoeff * faceArea``) and a matching source term.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["FixedValueBC"]


@BoundaryCondition.register("fixedValue")
class FixedValueBC(BoundaryCondition):
    """Fixed-value boundary condition.

    Prescribes a value at each boundary face.  The value can be uniform
    (single scalar) or non-uniform (per-face tensor).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._value = self._resolve_value()

    def _resolve_value(self) -> torch.Tensor:
        """Parse the ``value`` coefficient into a tensor."""
        raw = self._coeffs.get("value", 0.0)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        # Uniform scalar → broadcast to n_faces
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    @property
    def value(self) -> torch.Tensor:
        """Return the prescribed boundary values."""
        return self._value

    @value.setter
    def value(self, new_value: float | torch.Tensor) -> None:
        """Update the prescribed boundary values."""
        if isinstance(new_value, torch.Tensor):
            self._value = new_value.to(
                dtype=get_default_dtype(), device=get_device()
            )
        else:
            self._value = torch.full_like(self._patch.face_areas, float(new_value))

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face values to the prescribed value."""
        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = self._value
        else:
            field[self._patch.face_indices] = self._value
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method: large diagonal + matching source.

        For each boundary face adjacent to cell *c*:
            diag[c]   += deltaCoeff * faceArea
            source[c] += deltaCoeff * faceArea * prescribedValue
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
        values = self._value.to(device=device, dtype=dtype)

        # Implicit coefficient = deltaCoeff * area
        coeff = deltas * areas

        # Scatter-add into diagonal and source
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * values)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
