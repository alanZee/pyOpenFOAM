"""
Surface fields — face-centred field classes.

- :class:`surfaceScalarField` — scalar per face, shape ``(n_faces,)``
- :class:`surfaceVectorField` — 3-vector per face, shape ``(n_faces, 3)``

Surface fields cover **all** faces (internal + boundary).  The first
``n_internal_faces`` entries correspond to internal faces; the remaining
entries are boundary faces.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.boundary.boundary_field import BoundaryField

from .dimensions import DimensionSet
from .geometric_field import GeometricField
from .field_arithmetic import FieldArithmeticMixin

__all__ = [
    "SurfaceField",
    "surfaceScalarField",
    "surfaceVectorField",
]


# ---------------------------------------------------------------------------
# Base surface field
# ---------------------------------------------------------------------------


class SurfaceField(FieldArithmeticMixin, GeometricField):
    """Base class for face-centred (surface) fields.

    Internal field shape depends on the concrete type:
    - scalar: ``(n_faces,)``
    - vector: ``(n_faces, 3)``
    """

    @property
    def n_faces(self) -> int:
        """Total number of faces (internal + boundary)."""
        return self._mesh.n_faces

    @property
    def n_internal_faces(self) -> int:
        """Number of internal faces."""
        return self._mesh.n_internal_faces

    @property
    def internal_faces(self) -> torch.Tensor:
        """Values on internal faces."""
        return self._internal[: self._mesh.n_internal_faces]

    @property
    def boundary_faces(self) -> torch.Tensor:
        """Values on boundary faces."""
        return self._internal[self._mesh.n_internal_faces :]


# ---------------------------------------------------------------------------
# Concrete surface fields
# ---------------------------------------------------------------------------


class surfaceScalarField(SurfaceField):
    """Face-centred scalar field.

    Internal field: ``(n_faces,)`` tensor covering all faces
    (internal + boundary).

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    name : str
        Field name (e.g. ``"phi"``).
    dimensions : DimensionSet, optional
        Physical dimensions. Defaults to dimless.
    internal : torch.Tensor | float | None, optional
        Initial values for all faces. Defaults to zeros.
    boundary : BoundaryField, optional
        Boundary conditions.
    """

    def _expected_shape(self) -> tuple[int, ...]:
        return (self._mesh.n_faces,)


class surfaceVectorField(SurfaceField):
    """Face-centred vector field.

    Internal field: ``(n_faces, 3)`` tensor covering all faces.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    name : str
        Field name.
    dimensions : DimensionSet, optional
        Physical dimensions. Defaults to dimless.
    internal : torch.Tensor | float | None, optional
        Initial values for all faces. Defaults to zeros.
    boundary : BoundaryField, optional
        Boundary conditions.
    """

    def _expected_shape(self) -> tuple[int, ...]:
        return (self._mesh.n_faces, 3)
