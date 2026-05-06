"""
pyfoam.fields — Geometric field hierarchy for CFD.

Provides:

- :class:`DimensionSet` — 7-element physical dimension system
- :class:`GeometricField` — abstract base for all fields
- :class:`volScalarField` — cell-centred scalar
- :class:`volVectorField` — cell-centred vector
- :class:`volTensorField` — cell-centred tensor
- :class:`surfaceScalarField` — face-centred scalar
- :class:`surfaceVectorField` — face-centred vector
- :class:`FieldArithmeticMixin` — arithmetic operator overloads

All fields use PyTorch tensors and respect the global device/dtype
configuration from :mod:`pyfoam.core`.
"""

from pyfoam.fields.dimensions import DimensionSet, DimensionError
from pyfoam.fields.geometric_field import GeometricField
from pyfoam.fields.field_arithmetic import FieldArithmeticMixin
from pyfoam.fields.vol_fields import (
    VolField,
    volScalarField,
    volVectorField,
    volTensorField,
)
from pyfoam.fields.surface_fields import (
    SurfaceField,
    surfaceScalarField,
    surfaceVectorField,
)

__all__ = [
    # Dimensions
    "DimensionSet",
    "DimensionError",
    # Base
    "GeometricField",
    "FieldArithmeticMixin",
    # Volume fields
    "VolField",
    "volScalarField",
    "volVectorField",
    "volTensorField",
    # Surface fields
    "SurfaceField",
    "surfaceScalarField",
    "surfaceVectorField",
]
