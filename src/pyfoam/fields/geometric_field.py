"""
GeometricField — generic base class for all CFD fields.

Mirrors OpenFOAM's ``GeometricField<Type, PatchField, GeoMesh>`` template.
A field stores:

- **internalField** — tensor of values at cell centres (vol) or face centres (surface)
- **boundaryField** — :class:`~pyfoam.boundary.BoundaryField` collection of BCs
- **dimensions** — :class:`DimensionSet` for dimensional consistency checking
- **mesh** — reference to the mesh this field lives on

Arithmetic is mixed in via :class:`FieldArithmeticMixin` (defined in
``field_arithmetic.py``) so that concrete subclasses get ``+``, ``-``,
``*``, ``/`` with automatic dimension checking.

All tensors respect the global device/dtype from :mod:`pyfoam.core`.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.boundary.boundary_field import BoundaryField

from .dimensions import DimensionSet, DimensionError

__all__ = ["GeometricField"]


class GeometricField:
    """Abstract base for all geometric fields (vol, surface).

    Subclasses must set ``_mesh`` and ``_internal`` in ``__init__`` and
    implement :meth:`_n_values` (number of internal values expected).

    Parameters
    ----------
    mesh : Any
        The mesh this field belongs to (FvMesh, PolyMesh, etc.).
    name : str
        Field name (e.g. ``"p"``, ``"U"``).
    dimensions : DimensionSet
        Physical dimensions of the field.
    internal : torch.Tensor | float | None
        Initial internal field values.  If ``None``, initialised to zero.
    boundary : BoundaryField | None
        Boundary conditions.  If ``None``, an empty :class:`BoundaryField` is created.
    """

    def __init__(
        self,
        mesh: Any,
        name: str,
        dimensions: DimensionSet | None = None,
        internal: torch.Tensor | float | None = None,
        boundary: BoundaryField | None = None,
    ) -> None:
        self._mesh = mesh
        self._name = name
        self._dimensions = dimensions or DimensionSet.dimless()

        device = get_device()
        dtype = get_default_dtype()

        # Resolve internal field
        if internal is None:
            self._internal = torch.zeros(
                self._expected_shape(), device=device, dtype=dtype
            )
        elif isinstance(internal, (int, float)):
            self._internal = torch.full(
                self._expected_shape(), float(internal), device=device, dtype=dtype
            )
        else:
            # Tensor — ensure correct shape, device, dtype
            self._internal = internal.to(device=device, dtype=dtype)
            expected = self._expected_shape()
            if tuple(self._internal.shape) != tuple(expected):
                raise ValueError(
                    f"Internal field has shape {tuple(self._internal.shape)}, "
                    f"expected {tuple(expected)}"
                )

        # Boundary field
        self._boundary = boundary if boundary is not None else BoundaryField()

    # ------------------------------------------------------------------
    # Abstract interface (subclasses must implement)
    # ------------------------------------------------------------------

    def _expected_shape(self) -> tuple[int, ...]:
        """Return the expected shape of the internal field tensor.

        Override in subclasses:
        - volScalarField → (n_cells,)
        - volVectorField → (n_cells, 3)
        - volTensorField → (n_cells, 3, 3)
        - surfaceScalarField → (n_faces,)
        - surfaceVectorField → (n_faces, 3)
        """
        raise NotImplementedError(
            "Subclasses must implement _expected_shape()"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Field name."""
        return self._name

    @property
    def dimensions(self) -> DimensionSet:
        """Physical dimensions."""
        return self._dimensions

    @property
    def internal_field(self) -> torch.Tensor:
        """Internal field values tensor."""
        return self._internal

    @internal_field.setter
    def internal_field(self, value: torch.Tensor) -> None:
        """Set internal field values (with device/dtype enforcement)."""
        device = get_device()
        dtype = get_default_dtype()
        self._internal = value.to(device=device, dtype=dtype)

    @property
    def boundary_field(self) -> BoundaryField:
        """Boundary field collection."""
        return self._boundary

    @property
    def mesh(self) -> Any:
        """The mesh this field lives on."""
        return self._mesh

    @property
    def device(self) -> torch.device:
        """Device of the internal field tensor."""
        return self._internal.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the internal field tensor."""
        return self._internal.dtype

    # ------------------------------------------------------------------
    # Field assignment with BC enforcement
    # ------------------------------------------------------------------

    def assign(self, values: torch.Tensor | float) -> None:
        """Assign new values to the internal field.

        For vol fields, boundary conditions are applied after assignment
        so that boundary-face values are consistent with the new internal
        values and the prescribed BCs.

        Args:
            values: New values for the internal field.  If scalar, broadcast
                to all internal values.

        Raises:
            ValueError: If tensor shape does not match internal field.
        """
        device = get_device()
        dtype = get_default_dtype()

        if isinstance(values, (int, float)):
            self._internal.fill_(float(values))
        else:
            if tuple(values.shape) != tuple(self._internal.shape):
                raise ValueError(
                    f"Shape mismatch: got {tuple(values.shape)}, "
                    f"expected {tuple(self._internal.shape)}"
                )
            self._internal = values.to(device=device, dtype=dtype)

        # Apply boundary conditions if the boundary field has entries.
        # BCs operate on a concatenated [internal | boundary] tensor in
        # OpenFOAM, but here we only apply them when the boundary field
        # is non-empty and the face indices are valid for the internal
        # field tensor.  For surface fields the boundary faces are already
        # part of the internal tensor, so BC application is a no-op here.
        if len(self._boundary) > 0:
            self._boundary.apply(self._internal)

    # ------------------------------------------------------------------
    # Dimension checking helpers
    # ------------------------------------------------------------------

    def _check_same_dimensions(self, other: "GeometricField") -> None:
        """Raise if *other* has different dimensions (for add/sub)."""
        if self._dimensions != other._dimensions:
            raise DimensionError(
                f"Cannot operate on fields with different dimensions: "
                f"{self._name} {self._dimensions} and "
                f"{other._name} {other._dimensions}"
            )

    def _check_same_mesh(self, other: "GeometricField") -> None:
        """Raise if *other* lives on a different mesh."""
        if self._mesh is not other._mesh:
            raise ValueError(
                f"Fields '{self._name}' and '{other._name}' "
                f"are on different meshes"
            )

    # ------------------------------------------------------------------
    # Device / dtype transfer
    # ------------------------------------------------------------------

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "GeometricField":
        """Return a copy of this field on the given device/dtype.

        Does NOT modify in-place — returns a new field instance.
        """
        new_internal = self._internal
        if device is not None:
            new_internal = new_internal.to(device=device)
        if dtype is not None:
            new_internal = new_internal.to(dtype=dtype)

        return self.__class__(
            mesh=self._mesh,
            name=self._name,
            dimensions=self._dimensions,
            internal=new_internal,
            boundary=self._boundary,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self._name}', "
            f"dimensions={self._dimensions}, "
            f"shape={tuple(self._internal.shape)}, "
            f"device={self.device}, "
            f"dtype={self.dtype})"
        )
