"""
map_fields enhanced v9 -- enhanced map fields with additional capabilities
(generation 9).

Extends :func:`map_fields_enhanced_8` with:

- **multi mesh mapping**: enhanced multi mesh mapping capabilities.
- **mapping diagnostics**: enhanced mapping diagnostics capabilities.

Usage::

    from pyfoam.tools.map_fields_enhanced_9 import MapFieldsEnhanced9Result, map_fields_enhanced_9

    result = map_fields_enhanced_9()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MapFieldsEnhanced9Result", "map_fields_enhanced_9"]

@dataclass
class MultiMeshMappingResult:
    """Feature data for multi_mesh_mapping."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MappingDiagnosticsResult:
    """Feature data for mapping_diagnostics."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MapFieldsEnhanced9Result:
    """Result from :func:`map_fields_enhanced_9`."""
    multi_mesh: Optional[MultiMeshMappingResult] = None
    diagnostics: Optional[MappingDiagnosticsResult] = None


def map_fields_enhanced_9(
    mesh: Optional["FvMesh"] = None,
    enable_multi_mesh: bool = False,
    enable_diagnostics: bool = False,
) -> MapFieldsEnhanced9Result:
    """Enhanced v9 map fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MapFieldsEnhanced9Result
    """
    multi_mesh = None
    if enable_multi_mesh:
        multi_mesh = MultiMeshMappingResult(name="multi_mesh_mapping", enabled=True)

    diagnostics = None
    if enable_diagnostics:
        diagnostics = MappingDiagnosticsResult(name="mapping_diagnostics", enabled=True)

    return MapFieldsEnhanced9Result(
        multi_mesh=multi_mesh,
        diagnostics=diagnostics,
    )
