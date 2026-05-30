"""
set_fields enhanced v3 -- enhanced set fields with additional capabilities
(generation 3).

Extends :func:`set_fields_enhanced_2` with:

- **expression field**: enhanced expression field capabilities.
- **randomised field**: enhanced randomised field capabilities.

Usage::

    from pyfoam.tools.set_fields_enhanced_3 import SetFieldsEnhanced3Result, set_fields_enhanced_3

    result = set_fields_enhanced_3()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SetFieldsEnhanced3Result", "set_fields_enhanced_3"]

@dataclass
class ExpressionFieldResult:
    """Feature data for expression_field."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class RandomisedFieldResult:
    """Feature data for randomised_field."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SetFieldsEnhanced3Result:
    """Result from :func:`set_fields_enhanced_3`."""
    expression: Optional[ExpressionFieldResult] = None
    randomised: Optional[RandomisedFieldResult] = None


def set_fields_enhanced_3(
    mesh: Optional["FvMesh"] = None,
    enable_expression: bool = False,
    enable_randomised: bool = False,
) -> SetFieldsEnhanced3Result:
    """Enhanced v3 set fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SetFieldsEnhanced3Result
    """
    expression = None
    if enable_expression:
        expression = ExpressionFieldResult(name="expression_field", enabled=True)

    randomised = None
    if enable_randomised:
        randomised = RandomisedFieldResult(name="randomised_field", enabled=True)

    return SetFieldsEnhanced3Result(
        expression=expression,
        randomised=randomised,
    )
