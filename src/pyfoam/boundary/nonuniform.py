"""
nonUniform boundary condition.

Per-face values read from a file or explicitly provided as a tensor.
In OpenFOAM syntax::

    type    nonUniform;
    value   nonuniform (1 2 3 ...);

Each boundary face receives its own prescribed value.
The matrix contribution uses the penalty method, same as fixedValue.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["NonUniformBC"]


@BoundaryCondition.register("nonUniform")
class NonUniformBC(BoundaryCondition):
    """Non-uniform boundary condition.

    Prescribes per-face values at each boundary face.  The values
    are stored as a tensor with shape ``(n_faces,)``.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._value = self._resolve_value()

    def _resolve_value(self) -> torch.Tensor:
        """Parse the ``value`` coefficient into a per-face tensor."""
        raw = self._coeffs.get("value", None)
        device = get_device()
        dtype = get_default_dtype()
        n_faces = self._patch.n_faces

        if raw is None:
            return torch.zeros(n_faces, dtype=dtype, device=device)

        if isinstance(raw, torch.Tensor):
            t = raw.to(dtype=dtype, device=device)
            if t.shape[0] != n_faces:
                raise ValueError(
                    f"NonUniformBC value tensor has {t.shape[0]} elements "
                    f"but patch has {n_faces} faces."
                )
            return t

        if isinstance(raw, (list, tuple)):
            t = torch.tensor(raw, dtype=dtype, device=device)
            if t.shape[0] != n_faces:
                raise ValueError(
                    f"NonUniformBC value sequence has {t.shape[0]} elements "
                    f"but patch has {n_faces} faces."
                )
            return t

        # Scalar fallback: broadcast to all faces
        return torch.full((n_faces,), float(raw), dtype=dtype, device=device)

    @property
    def value(self) -> torch.Tensor:
        """Return the prescribed per-face values."""
        return self._value

    @value.setter
    def value(self, new_value: torch.Tensor | list | tuple) -> None:
        """Update the per-face values."""
        device = get_device()
        dtype = get_default_dtype()
        if isinstance(new_value, torch.Tensor):
            self._value = new_value.to(dtype=dtype, device=device)
        else:
            self._value = torch.tensor(new_value, dtype=dtype, device=device)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face values from the stored tensor."""
        values = self._value.to(device=field.device, dtype=field.dtype)
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
        values = self._value.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * values)

        return diag, source
