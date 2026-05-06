"""
Base boundary condition class with RTS (Run-Time Selection) registry.

In OpenFOAM, boundary conditions are selected at run-time from a dictionary
via the ``type`` keyword.  This module provides an equivalent mechanism in
Python using a class-level registry and decorator pattern.

Usage::

    @BoundaryCondition.register("fixedValue")
    class FixedValueBC(BoundaryCondition):
        ...

    # Factory creation
    bc = BoundaryCondition.create("fixedValue", patch, coeffs={"value": 1.0})
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Type

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "BoundaryCondition",
    "Patch",
]


# ---------------------------------------------------------------------------
# Patch data structure
# ---------------------------------------------------------------------------


@dataclass
class Patch:
    """Lightweight description of a boundary patch.

    Holds the geometry and connectivity needed by boundary conditions.
    This is *not* a full mesh class (that is Task 2) — just enough
    information for BCs to operate.

    Attributes:
        name: Patch name (e.g. ``"inlet"``, ``"wall"``).
        face_indices: 1-D integer tensor of face indices belonging to this patch.
        face_normals: ``(n_faces, 3)`` tensor of outward-pointing unit normals.
        face_areas: ``(n_faces,)`` tensor of face areas.
        delta_coeffs: ``(n_faces,)`` tensor of 1/distance coefficients
            (distance from cell-centre to face-centre).  Used for implicit
            BC treatment.
        owner_cells: ``(n_faces,)`` tensor of cell indices adjacent to the patch.
        neighbour_patch: Name of the coupled patch (for cyclic BCs).  ``None``
            for uncoupled patches.
    """

    name: str
    face_indices: torch.Tensor
    face_normals: torch.Tensor
    face_areas: torch.Tensor
    delta_coeffs: torch.Tensor
    owner_cells: torch.Tensor
    neighbour_patch: str | None = None

    @property
    def n_faces(self) -> int:
        """Number of faces in the patch."""
        return int(self.face_indices.shape[0])

    def to(self, device: torch.device | str | None = None) -> Patch:
        """Return a copy with all tensors moved to *device*."""
        if device is None:
            return self
        return Patch(
            name=self.name,
            face_indices=self.face_indices.to(device=device),
            face_normals=self.face_normals.to(device=device),
            face_areas=self.face_areas.to(device=device),
            delta_coeffs=self.delta_coeffs.to(device=device),
            owner_cells=self.owner_cells.to(device=device),
            neighbour_patch=self.neighbour_patch,
        )


# ---------------------------------------------------------------------------
# BoundaryCondition base class
# ---------------------------------------------------------------------------


class BoundaryCondition(ABC):
    """Abstract base class for all boundary conditions.

    Subclasses must implement :meth:`apply` and :meth:`matrix_contributions`.

    RTS (Run-Time Selection) registry allows string-based lookup::

        @BoundaryCondition.register("fixedValue")
        class FixedValueBC(BoundaryCondition):
            ...

        bc = BoundaryCondition.create("fixedValue", patch, coeffs)
    """

    # Class-level RTS registry: name -> class
    _registry: ClassVar[dict[str, Type[BoundaryCondition]]] = {}

    def __init__(
        self,
        patch: Patch,
        coeffs: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the boundary condition.

        Args:
            patch: Boundary patch geometry and connectivity.
            coeffs: Dictionary of BC coefficients (from OpenFOAM dict).
        """
        self._patch = patch
        self._coeffs: dict[str, Any] = coeffs or {}

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a BC class under *name*.

        Usage::

            @BoundaryCondition.register("fixedValue")
            class FixedValueBC(BoundaryCondition):
                ...
        """

        def decorator(bc_cls: Type[BoundaryCondition]) -> Type[BoundaryCondition]:
            if name in cls._registry:
                raise ValueError(
                    f"Boundary condition '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = bc_cls
            return bc_cls

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        patch: Patch,
        coeffs: dict[str, Any] | None = None,
    ) -> BoundaryCondition:
        """Factory: create a BC instance by registered *name*.

        Args:
            name: Registered BC type name (e.g. ``"fixedValue"``).
            patch: Boundary patch.
            coeffs: BC coefficients.

        Returns:
            Instantiated boundary condition.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown boundary condition type '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](patch, coeffs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered BC type names."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def patch(self) -> Patch:
        """Return the bound patch."""
        return self._patch

    @property
    def coeffs(self) -> dict[str, Any]:
        """Return the BC coefficient dictionary."""
        return self._coeffs

    @property
    def type_name(self) -> str:
        """Return the registered type name for this BC class."""
        for name, bc_cls in self._registry.items():
            if isinstance(self, bc_cls):
                return name
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply the BC to *field*, modifying boundary-face values in-place.

        Args:
            field: Full field tensor (one value per cell or per face).
                For cell-centred fields the boundary faces are at the end
                of the tensor (standard OpenFOAM layout).
            patch_idx: Optional explicit start index into *field* for this
                patch.  When ``None``, the patch's ``face_indices`` are used.

        Returns:
            The (possibly modified) field tensor.
        """

    @abstractmethod
    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fvMatrix diagonal and source contributions.

        In FVM, boundary conditions contribute to the linear system
        ``A φ = b`` through additional diagonal (``aP``) and source
        (``bP``) terms for cells adjacent to the boundary.

        Args:
            field: Current field values.
            n_cells: Total number of cells (size of the linear system).
            diag: Pre-existing diagonal tensor to accumulate into.
                Created as zeros if ``None``.
            source: Pre-existing source tensor to accumulate into.
                Created as zeros if ``None``.

        Returns:
            ``(diag, source)`` tuple — the accumulated diagonal and
            source tensors of shape ``(n_cells,)``.
        """

    # ------------------------------------------------------------------
    # Utility helpers for subclasses
    # ------------------------------------------------------------------

    def _get_uniform_value(self, key: str = "value", default: float = 0.0) -> torch.Tensor:
        """Extract a uniform value from coefficients as a scalar tensor."""
        val = self._coeffs.get(key, default)
        if isinstance(val, torch.Tensor):
            return val.to(dtype=get_default_dtype(), device=get_device())
        return torch.tensor(val, dtype=get_default_dtype(), device=get_device())

    def _get_field_values(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
    ) -> torch.Tensor:
        """Return the slice of *field* corresponding to this patch's faces."""
        if patch_idx is not None:
            n = self._patch.n_faces
            return field[patch_idx : patch_idx + n]
        return field[self._patch.face_indices]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(patch='{self._patch.name}', "
            f"n_faces={self._patch.n_faces})"
        )
