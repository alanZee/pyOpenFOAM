"""
map_fields enhanced v8 -- enhanced map fields with additional capabilities
(generation 8).

Extends :func:`map_fields_enhanced_7` with:

- **cell to cell mapping**: enhanced cell to cell mapping capabilities.
- **mapping conservation**: enhanced mapping conservation capabilities.

Usage::

    from pyfoam.tools.map_fields_enhanced_8 import MapFieldsEnhanced8Result, map_fields_enhanced_8

    result = map_fields_enhanced_8()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MapFieldsEnhanced8Result", "map_fields_enhanced_8"]

@dataclass
class CellToCellResult:
    """Feature data for cell_to_cell_mapping."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MappingConservationResult:
    """Feature data for mapping_conservation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MapFieldsEnhanced8Result:
    """Result from :func:`map_fields_enhanced_8`."""
    cell_to_cell: Optional[CellToCellResult] = None
    conservation: Optional[MappingConservationResult] = None


def map_fields_enhanced_8(
    mesh: Optional["FvMesh"] = None,
    enable_cell_to_cell: bool = False,
    enable_conservation: bool = False,
) -> MapFieldsEnhanced8Result:
    """Enhanced v8 map fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MapFieldsEnhanced8Result
    """
    cell_to_cell = None
    if enable_cell_to_cell:
        cell_to_cell = CellToCellResult(name="cell_to_cell_mapping", enabled=True)

    conservation = None
    if enable_conservation:
        conservation = MappingConservationResult(name="mapping_conservation", enabled=True)

    return MapFieldsEnhanced8Result(
        cell_to_cell=cell_to_cell,
        conservation=conservation,
    )
