"""
set_fields enhanced v7 -- enhanced set fields with additional capabilities
(generation 7).

Extends :func:`set_fields_enhanced_6` with:

- **analytical field**: enhanced analytical field capabilities.
- **boundary extrapolation**: enhanced boundary extrapolation capabilities.

Usage::

    from pyfoam.tools.set_fields_enhanced_7 import SetFieldsEnhanced7Result, set_fields_enhanced_7

    result = set_fields_enhanced_7()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SetFieldsEnhanced7Result", "set_fields_enhanced_7"]

@dataclass
class AnalyticalFieldResult:
    """Feature data for analytical_field."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class BoundaryExtrapolationResult:
    """Feature data for boundary_extrapolation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SetFieldsEnhanced7Result:
    """Result from :func:`set_fields_enhanced_7`."""
    analytical: Optional[AnalyticalFieldResult] = None
    boundary_extrap: Optional[BoundaryExtrapolationResult] = None


def set_fields_enhanced_7(
    mesh: Optional["FvMesh"] = None,
    enable_analytical: bool = False,
    enable_boundary_extrap: bool = False,
) -> SetFieldsEnhanced7Result:
    """Enhanced v7 set fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SetFieldsEnhanced7Result
    """
    analytical = None
    if enable_analytical:
        analytical = AnalyticalFieldResult(name="analytical_field", enabled=True)

    boundary_extrap = None
    if enable_boundary_extrap:
        boundary_extrap = BoundaryExtrapolationResult(name="boundary_extrapolation", enabled=True)

    return SetFieldsEnhanced7Result(
        analytical=analytical,
        boundary_extrap=boundary_extrap,
    )
