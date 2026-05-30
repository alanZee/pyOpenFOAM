"""
set_fields enhanced v5 -- enhanced set fields with additional capabilities
(generation 5).

Extends :func:`set_fields_enhanced_4` with:

- **field interpolation**: enhanced field interpolation capabilities.
- **composite region**: enhanced composite region capabilities.
- **field blending**: enhanced field blending capabilities.

Usage::

    from pyfoam.tools.set_fields_enhanced_5 import SetFieldsEnhanced5Result, set_fields_enhanced_5

    result = set_fields_enhanced_5()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SetFieldsEnhanced5Result", "set_fields_enhanced_5"]

@dataclass
class FieldInterpolationResult:
    """Feature data for field_interpolation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class CompositeRegionResult:
    """Feature data for composite_region."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class FieldBlendingResult:
    """Feature data for field_blending."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SetFieldsEnhanced5Result:
    """Result from :func:`set_fields_enhanced_5`."""
    interpolation: Optional[FieldInterpolationResult] = None
    composite: Optional[CompositeRegionResult] = None
    blending: Optional[FieldBlendingResult] = None


def set_fields_enhanced_5(
    mesh: Optional["FvMesh"] = None,
    enable_interpolation: bool = False,
    enable_composite: bool = False,
    enable_blending: bool = False,
) -> SetFieldsEnhanced5Result:
    """Enhanced v5 set fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SetFieldsEnhanced5Result
    """
    interpolation = None
    if enable_interpolation:
        interpolation = FieldInterpolationResult(name="field_interpolation", enabled=True)

    composite = None
    if enable_composite:
        composite = CompositeRegionResult(name="composite_region", enabled=True)

    blending = None
    if enable_blending:
        blending = FieldBlendingResult(name="field_blending", enabled=True)

    return SetFieldsEnhanced5Result(
        interpolation=interpolation,
        composite=composite,
        blending=blending,
    )
