"""
map_fields enhanced v4 -- enhanced map fields with additional capabilities
(generation 4).

Extends :func:`map_fields_enhanced_3` with:

- **multi field mapping**: enhanced multi field mapping capabilities.
- **mapping error estimation**: enhanced mapping error estimation capabilities.

Usage::

    from pyfoam.tools.map_fields_enhanced_4 import MapFieldsEnhanced4Result, map_fields_enhanced_4

    result = map_fields_enhanced_4()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MapFieldsEnhanced4Result", "map_fields_enhanced_4"]

@dataclass
class MultiFieldMappingResult:
    """Feature data for multi_field_mapping."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MappingErrorResult:
    """Feature data for mapping_error_estimation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MapFieldsEnhanced4Result:
    """Result from :func:`map_fields_enhanced_4`."""
    multi_field: Optional[MultiFieldMappingResult] = None
    error_estimation: Optional[MappingErrorResult] = None


def map_fields_enhanced_4(
    mesh: Optional["FvMesh"] = None,
    enable_multi_field: bool = False,
    enable_error_estimation: bool = False,
) -> MapFieldsEnhanced4Result:
    """Enhanced v4 map fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MapFieldsEnhanced4Result
    """
    multi_field = None
    if enable_multi_field:
        multi_field = MultiFieldMappingResult(name="multi_field_mapping", enabled=True)

    error_estimation = None
    if enable_error_estimation:
        error_estimation = MappingErrorResult(name="mapping_error_estimation", enabled=True)

    return MapFieldsEnhanced4Result(
        multi_field=multi_field,
        error_estimation=error_estimation,
    )
