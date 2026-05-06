"""
Field arithmetic — operator overloads for GeometricField.

Provides a mixin class that adds ``+``, ``-``, ``*``, ``/`` to any
:class:`GeometricField` subclass with automatic dimensional consistency
checking.

Supported operations
--------------------
- **field + field** → same dimensions, element-wise internal addition
- **field - field** → same dimensions, element-wise internal subtraction
- **field * scalar** → scalar broadcast multiplication (scalar is dimensionless)
- **scalar * field** → reverse scalar multiplication
- **field * field** → element-wise, dimensions are summed
- **field / field** → element-wise, dimensions are subtracted
- **field / scalar** → scalar broadcast division
- **-field** → negate internal values

All operations return a **new** field of the same concrete type.
Boundary conditions are **not** carried to the result (result has empty BCs).
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.boundary.boundary_field import BoundaryField

from .dimensions import DimensionSet, DimensionError

__all__ = ["FieldArithmeticMixin"]


def _is_field(obj: Any) -> bool:
    """Check if *obj* is a GeometricField (duck-typed, avoids circular import)."""
    return hasattr(obj, "_internal") and hasattr(obj, "_dimensions") and hasattr(obj, "_mesh")


class FieldArithmeticMixin:
    """Mixin that adds arithmetic operators to GeometricField subclasses.

    Must be used in combination with :class:`GeometricField` (the mixin
    accesses ``self._internal``, ``self._dimensions``, ``self._mesh``,
    ``self._name``, etc.).
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_field(
        self,
        internal: torch.Tensor,
        dimensions: DimensionSet,
        name: str = "",
    ) -> "FieldArithmeticMixin":
        """Create a new field of the same concrete type with given data.

        Boundary conditions are reset to empty.
        """
        return self.__class__(
            mesh=self._mesh,
            name=name or f"({self._name})",
            dimensions=dimensions,
            internal=internal,
            boundary=BoundaryField(),
        )

    # ------------------------------------------------------------------
    # Addition
    # ------------------------------------------------------------------

    def __add__(self, other: Any) -> "FieldArithmeticMixin":
        """field + field (same dimensions) or field + scalar (dimless scalar)."""
        if _is_field(other):
            self._check_same_dimensions(other)
            self._check_same_mesh(other)
            return self._new_field(
                self._internal + other._internal,
                self._dimensions,
            )
        if isinstance(other, (int, float)):
            # Adding a scalar only valid if field is dimensionless
            if not self._dimensions.is_dimless:
                raise DimensionError(
                    f"Cannot add scalar to dimensional field {self._name} "
                    f"{self._dimensions}"
                )
            return self._new_field(
                self._internal + other,
                self._dimensions,
            )
        return NotImplemented

    def __radd__(self, other: Any) -> "FieldArithmeticMixin":
        """scalar + field."""
        return self.__add__(other)

    # ------------------------------------------------------------------
    # Subtraction
    # ------------------------------------------------------------------

    def __sub__(self, other: Any) -> "FieldArithmeticMixin":
        """field - field (same dimensions) or field - scalar (dimless scalar)."""
        if _is_field(other):
            self._check_same_dimensions(other)
            self._check_same_mesh(other)
            return self._new_field(
                self._internal - other._internal,
                self._dimensions,
            )
        if isinstance(other, (int, float)):
            if not self._dimensions.is_dimless:
                raise DimensionError(
                    f"Cannot subtract scalar from dimensional field {self._name} "
                    f"{self._dimensions}"
                )
            return self._new_field(
                self._internal - other,
                self._dimensions,
            )
        return NotImplemented

    def __rsub__(self, other: Any) -> "FieldArithmeticMixin":
        """scalar - field."""
        if isinstance(other, (int, float)):
            if not self._dimensions.is_dimless:
                raise DimensionError(
                    f"Cannot subtract dimensional field {self._name} "
                    f"{self._dimensions} from scalar"
                )
            return self._new_field(
                other - self._internal,
                self._dimensions,
            )
        return NotImplemented

    # ------------------------------------------------------------------
    # Multiplication
    # ------------------------------------------------------------------

    def __mul__(self, other: Any) -> "FieldArithmeticMixin":
        """field * field (element-wise, dims added) or field * scalar."""
        if _is_field(other):
            self._check_same_mesh(other)
            new_dims = self._dimensions * other._dimensions
            return self._new_field(
                self._internal * other._internal,
                new_dims,
            )
        if isinstance(other, (int, float)):
            return self._new_field(
                self._internal * other,
                self._dimensions,
            )
        if isinstance(other, torch.Tensor):
            return self._new_field(
                self._internal * other,
                self._dimensions,
            )
        return NotImplemented

    def __rmul__(self, other: Any) -> "FieldArithmeticMixin":
        """scalar * field or tensor * field."""
        if isinstance(other, (int, float, torch.Tensor)):
            return self.__mul__(other)
        return NotImplemented

    # ------------------------------------------------------------------
    # Division
    # ------------------------------------------------------------------

    def __truediv__(self, other: Any) -> "FieldArithmeticMixin":
        """field / field (element-wise, dims subtracted) or field / scalar."""
        if _is_field(other):
            self._check_same_mesh(other)
            new_dims = self._dimensions / other._dimensions
            return self._new_field(
                self._internal / other._internal,
                new_dims,
            )
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide field by zero")
            return self._new_field(
                self._internal / other,
                self._dimensions,
            )
        return NotImplemented

    def __rtruediv__(self, other: Any) -> "FieldArithmeticMixin":
        """scalar / field."""
        if isinstance(other, (int, float)):
            new_dims = DimensionSet.dimless() / self._dimensions
            return self._new_field(
                other / self._internal,
                new_dims,
            )
        return NotImplemented

    # ------------------------------------------------------------------
    # Unary negation
    # ------------------------------------------------------------------

    def __neg__(self) -> "FieldArithmeticMixin":
        """-field."""
        return self._new_field(
            -self._internal,
            self._dimensions,
        )

    # ------------------------------------------------------------------
    # In-place operations (modify internal field, keep dimensions)
    # ------------------------------------------------------------------

    def __iadd__(self, other: Any) -> "FieldArithmeticMixin":
        """field += field or field += scalar."""
        if _is_field(other):
            self._check_same_dimensions(other)
            self._check_same_mesh(other)
            self._internal = self._internal + other._internal
            return self
        if isinstance(other, (int, float)):
            if not self._dimensions.is_dimless:
                raise DimensionError(
                    f"Cannot add scalar to dimensional field {self._name}"
                )
            self._internal = self._internal + other
            return self
        return NotImplemented

    def __isub__(self, other: Any) -> "FieldArithmeticMixin":
        """field -= field or field -= scalar."""
        if _is_field(other):
            self._check_same_dimensions(other)
            self._check_same_mesh(other)
            self._internal = self._internal - other._internal
            return self
        if isinstance(other, (int, float)):
            if not self._dimensions.is_dimless:
                raise DimensionError(
                    f"Cannot subtract scalar from dimensional field {self._name}"
                )
            self._internal = self._internal - other
            return self
        return NotImplemented

    def __imul__(self, other: Any) -> "FieldArithmeticMixin":
        """field *= scalar."""
        if isinstance(other, (int, float)):
            self._internal = self._internal * other
            return self
        return NotImplemented

    def __itruediv__(self, other: Any) -> "FieldArithmeticMixin":
        """field /= scalar."""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide field by zero")
            self._internal = self._internal / other
            return self
        return NotImplemented
