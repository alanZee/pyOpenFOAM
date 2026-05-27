"""
generic boundary condition.

A general-purpose mixed BC that blends fixed-value (Dirichlet) and
zero-gradient (Neumann) treatment based on a configurable
``valueFraction``.

In OpenFOAM syntax::

    type           generic;
    value          uniform 10;
    gradient       uniform 0;
    valueFraction  uniform 0.8;

When ``valueFraction = 1``, the BC behaves as pure fixed value.
When ``valueFraction = 0``, it behaves as pure zero gradient.
Intermediate values produce a weighted blend.

The matrix contribution uses the penalty method with the blended
coefficient.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["GenericBC"]


@BoundaryCondition.register("generic")
class GenericBC(BoundaryCondition):
    """Generic mixed boundary condition.

    Blends fixed-value and zero-gradient treatment using
    ``valueFraction`` as the weighting factor:
        face_value = valueFraction * prescribed_value
                   + (1 - valueFraction) * owner_cell_value
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._value = self._resolve_tensor("value", 0.0)
        self._gradient = self._resolve_tensor("gradient", 0.0)
        self._value_fraction = self._resolve_tensor("valueFraction", 1.0)

    def _resolve_tensor(self, key: str, default: float) -> torch.Tensor:
        """Parse a coefficient into a per-face tensor."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.full(
            (self._patch.n_faces,),
            float(raw),
            dtype=get_default_dtype(),
            device=get_device(),
        )

    @property
    def value(self) -> torch.Tensor:
        """Return the prescribed fixed-value part."""
        return self._value

    @property
    def gradient(self) -> torch.Tensor:
        """Return the prescribed gradient part."""
        return self._gradient

    @property
    def value_fraction(self) -> torch.Tensor:
        """Return the value fraction (0 = pure gradient, 1 = pure value)."""
        return self._value_fraction

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Blend fixed value and zero gradient based on valueFraction.

        face_value = f * prescribed_value + (1 - f) * owner_value
        where f = valueFraction.
        """
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]
        f = self._value_fraction.to(device=field.device, dtype=field.dtype)
        v = self._value.to(device=field.device, dtype=field.dtype)

        face_values = f * v + (1.0 - f) * owner_values

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
        """Weighted penalty method based on valueFraction.

        For each boundary face adjacent to cell *c*:
            coeff        = deltaCoeff * faceArea * valueFraction
            diag[c]     += coeff
            source[c]   += coeff * prescribed_value
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
        f = self._value_fraction.to(device=device, dtype=dtype)

        # Implicit coefficient = deltaCoeff * area * valueFraction
        coeff = deltas * areas * f

        # Scatter-add into diagonal and source
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * values)

        return diag, source
