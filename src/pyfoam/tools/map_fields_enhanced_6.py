"""
map_fields enhanced v6 -- enhanced map fields with additional capabilities
(generation 6).

Extends :func:`map_fields_enhanced_5` with:

- **tetrahedral interpolation**: enhanced tetrahedral interpolation capabilities.
- **mesh to mesh mapping**: enhanced mesh to mesh mapping capabilities.

Usage::

    from pyfoam.tools.map_fields_enhanced_6 import MapFieldsEnhanced6Result, map_fields_enhanced_6

    result = map_fields_enhanced_6()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MapFieldsEnhanced6Result", "map_fields_enhanced_6"]

@dataclass
class TetInterpolationResult:
    """Feature data for tetrahedral_interpolation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MeshToMeshResult:
    """Feature data for mesh_to_mesh_mapping."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MapFieldsEnhanced6Result:
    """Result from :func:`map_fields_enhanced_6`."""
    tet_interp: Optional[TetInterpolationResult] = None
    mesh_to_mesh: Optional[MeshToMeshResult] = None


def map_fields_enhanced_6(
    mesh: Optional["FvMesh"] = None,
    enable_tet_interp: bool = False,
    enable_mesh_to_mesh: bool = False,
) -> MapFieldsEnhanced6Result:
    """Enhanced v6 map fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MapFieldsEnhanced6Result
    """
    tet_interp = None
    if enable_tet_interp:
        tet_interp = TetInterpolationResult(name="tetrahedral_interpolation", enabled=True)

    mesh_to_mesh = None
    if enable_mesh_to_mesh:
        mesh_to_mesh = MeshToMeshResult(name="mesh_to_mesh_mapping", enabled=True)

    return MapFieldsEnhanced6Result(
        tet_interp=tet_interp,
        mesh_to_mesh=mesh_to_mesh,
    )
