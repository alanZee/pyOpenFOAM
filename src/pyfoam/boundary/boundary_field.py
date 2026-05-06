"""
BoundaryField — collection of boundary conditions for a single field.

Manages one :class:`BoundaryCondition` per patch and provides a unified
interface for applying all BCs and computing matrix contributions.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["BoundaryField"]


class BoundaryField:
    """Collection of boundary conditions for all patches of a field.

    Stores one :class:`BoundaryCondition` per boundary patch and
    provides batch operations for applying BCs and computing
    fvMatrix contributions.

    Usage::

        bf = BoundaryField()
        bf.add(fixed_value_bc)
        bf.add(zero_gradient_bc)

        # Apply all BCs to the field
        bf.apply(field)

        # Compute matrix contributions
        diag, source = bf.matrix_contributions(field, n_cells=100)
    """

    def __init__(self) -> None:
        self._bcs: list[BoundaryCondition] = []
        self._name_to_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def add(self, bc: BoundaryCondition) -> None:
        """Add a boundary condition to the collection.

        Args:
            bc: Boundary condition instance.

        Raises:
            ValueError: If a BC for the same patch name already exists.
        """
        name = bc.patch.name
        if name in self._name_to_idx:
            raise ValueError(
                f"Boundary condition for patch '{name}' already exists "
                f"(type={self._bcs[self._name_to_idx[name]].type_name}). "
                f"Remove it first or use replace()."
            )
        self._name_to_idx[name] = len(self._bcs)
        self._bcs.append(bc)

    def remove(self, patch_name: str) -> BoundaryCondition:
        """Remove and return the BC for *patch_name*.

        Raises:
            KeyError: If no BC exists for *patch_name*.
        """
        if patch_name not in self._name_to_idx:
            raise KeyError(f"No boundary condition for patch '{patch_name}'")
        idx = self._name_to_idx.pop(patch_name)
        bc = self._bcs.pop(idx)
        # Rebuild index mapping
        self._name_to_idx = {
            bc_i.patch.name: i for i, bc_i in enumerate(self._bcs)
        }
        return bc

    def replace(self, bc: BoundaryCondition) -> BoundaryCondition | None:
        """Replace the BC for the same patch.  Returns old BC or None."""
        name = bc.patch.name
        old = None
        if name in self._name_to_idx:
            old = self.remove(name)
        self.add(bc)
        return old

    def get(self, patch_name: str) -> BoundaryCondition:
        """Return the BC for *patch_name*.

        Raises:
            KeyError: If no BC exists for *patch_name*.
        """
        if patch_name not in self._name_to_idx:
            raise KeyError(f"No boundary condition for patch '{patch_name}'")
        return self._bcs[self._name_to_idx[patch_name]]

    def __contains__(self, patch_name: str) -> bool:
        return patch_name in self._name_to_idx

    def __len__(self) -> int:
        return len(self._bcs)

    def __getitem__(self, idx: int) -> BoundaryCondition:
        return self._bcs[idx]

    def __iter__(self) -> Iterator[BoundaryCondition]:
        return iter(self._bcs)

    def __repr__(self) -> str:
        types = [bc.type_name for bc in self._bcs]
        return f"BoundaryField({len(self._bcs)} patches: {types})"

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """Apply all boundary conditions to *field*.

        Each BC modifies the boundary-face values of *field* in-place.
        BCs are applied in registration order.

        Args:
            field: Full field tensor.

        Returns:
            The modified field.
        """
        offset = self._compute_offset(field)
        for bc in self._bcs:
            patch_idx = offset.get(bc.patch.name)
            bc.apply(field, patch_idx=patch_idx)
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fvMatrix contributions from all boundary conditions.

        Args:
            field: Current field values.
            n_cells: Total number of cells.

        Returns:
            ``(diag, source)`` tensors of shape ``(n_cells,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        diag = torch.zeros(n_cells, device=device, dtype=dtype)
        source = torch.zeros(n_cells, device=device, dtype=dtype)

        for bc in self._bcs:
            bc.matrix_contributions(field, n_cells, diag=diag, source=source)

        return diag, source

    def total_n_faces(self) -> int:
        """Return total number of boundary faces across all patches."""
        return sum(bc.patch.n_faces for bc in self._bcs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_offset(field: torch.Tensor) -> dict[str, int | None]:
        """Compute the field-offset for each patch.

        In OpenFOAM, boundary-face values are stored after internal
        cell values.  This helper returns ``None`` for each patch
        (indicating that face_indices should be used instead).
        """
        # When using face_indices directly, offsets are not needed.
        return {}
