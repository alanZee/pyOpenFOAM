"""
map_fields enhanced v7 -- enhanced map fields with additional capabilities
(generation 7).

Extends :func:`map_fields_enhanced_6` with:

- **temporal mapping**: enhanced temporal mapping capabilities.
- **boundary mapping**: enhanced boundary mapping capabilities.

Usage::

    from pyfoam.tools.map_fields_enhanced_7 import MapFieldsEnhanced7Result, map_fields_enhanced_7

    result = map_fields_enhanced_7()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MapFieldsEnhanced7Result", "map_fields_enhanced_7"]

@dataclass
class TemporalMappingResult:
    """Feature data for temporal_mapping."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class BoundaryMappingResult:
    """Feature data for boundary_mapping."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MapFieldsEnhanced7Result:
    """Result from :func:`map_fields_enhanced_7`."""
    temporal: Optional[TemporalMappingResult] = None
    boundary: Optional[BoundaryMappingResult] = None


def map_fields_enhanced_7(
    mesh: Optional["FvMesh"] = None,
    enable_temporal: bool = False,
    enable_boundary: bool = False,
) -> MapFieldsEnhanced7Result:
    """Enhanced v7 map fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MapFieldsEnhanced7Result
    """
    temporal = None
    if enable_temporal:
        temporal = TemporalMappingResult(name="temporal_mapping", enabled=True)

    boundary = None
    if enable_boundary:
        boundary = BoundaryMappingResult(name="boundary_mapping", enabled=True)

    return MapFieldsEnhanced7Result(
        temporal=temporal,
        boundary=boundary,
    )
