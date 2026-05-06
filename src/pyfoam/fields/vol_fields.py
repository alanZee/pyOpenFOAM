"""
Volume fields — cell-centred field classes.

- :class:`volScalarField` — scalar per cell, shape ``(n_cells,)``
- :class:`volVectorField` — 3-vector per cell, shape ``(n_cells, 3)``
- :class:`volTensorField` — 3×3 tensor per cell, shape ``(n_cells, 3, 3)``

All inherit arithmetic from :class:`FieldArithmeticMixin` and dimensional
checking from :class:`GeometricField`.
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
    "VolField",
    "volScalarField",
    "volVectorField",
    "volTensorField",
]


# ---------------------------------------------------------------------------
# Base vol field
# ---------------------------------------------------------------------------


class VolField(FieldArithmeticMixin, GeometricField):
    """Base class for cell-centred (volume) fields.

    Internal field shape depends on the concrete type:
    - scalar: ``(n_cells,)``
    - vector: ``(n_cells, 3)``
    - tensor: ``(n_cells, 3, 3)``
    """

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self._mesh.n_cells


# ---------------------------------------------------------------------------
# Concrete volume fields
# ---------------------------------------------------------------------------


class volScalarField(VolField):
    """Cell-centred scalar field.

    Internal field: ``(n_cells,)`` tensor.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    name : str
        Field name (e.g. ``"p"``).
    dimensions : DimensionSet, optional
        Physical dimensions. Defaults to dimless.
    internal : torch.Tensor | float | None, optional
        Initial values. Defaults to zeros.
    boundary : BoundaryField, optional
        Boundary conditions.

    Examples::

        p = volScalarField(mesh, "p", dimensions=DimensionSet(mass=1, length=-1, time=-2))
        p.assign(torch.ones(mesh.n_cells) * 101325.0)
    """

    def _expected_shape(self) -> tuple[int, ...]:
        return (self._mesh.n_cells,)


class volVectorField(VolField):
    """Cell-centred vector field.

    Internal field: ``(n_cells, 3)`` tensor.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    name : str
        Field name (e.g. ``"U"``).
    dimensions : DimensionSet, optional
        Physical dimensions. Defaults to dimless.
    internal : torch.Tensor | float | None, optional
        Initial values. Defaults to zeros.
    boundary : BoundaryField, optional
        Boundary conditions.
    """

    def _expected_shape(self) -> tuple[int, ...]:
        return (self._mesh.n_cells, 3)


class volTensorField(VolField):
    """Cell-centred tensor field.

    Internal field: ``(n_cells, 3, 3)`` tensor.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    name : str
        Field name.
    dimensions : DimensionSet, optional
        Physical dimensions. Defaults to dimless.
    internal : torch.Tensor | float | None, optional
        Initial values. Defaults to zeros.
    boundary : BoundaryField, optional
        Boundary conditions.
    """

    def _expected_shape(self) -> tuple[int, ...]:
        return (self._mesh.n_cells, 3, 3)
