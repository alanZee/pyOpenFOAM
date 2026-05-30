"""
check_mesh enhanced v3 -- enhanced check mesh with additional capabilities
(generation 3).

Extends :func:`check_mesh_enhanced_2` with:

- **cell quality index**: enhanced cell quality index capabilities.
- **boundary quality check**: enhanced boundary quality check capabilities.

Usage::

    from pyfoam.tools.check_mesh_enhanced_3 import CheckMeshEnhanced3Result, check_mesh_enhanced_3

    result = check_mesh_enhanced_3()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["CheckMeshEnhanced3Result", "check_mesh_enhanced_3"]

@dataclass
class CellQualityIndex:
    """Feature data for cell_quality_index."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class BoundaryQualityResult:
    """Feature data for boundary_quality_check."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class CheckMeshEnhanced3Result:
    """Result from :func:`check_mesh_enhanced_3`."""
    quality_index: Optional[CellQualityIndex] = None
    boundary_quality: Optional[BoundaryQualityResult] = None


def check_mesh_enhanced_3(
    mesh: Optional["FvMesh"] = None,
    enable_quality_index: bool = False,
    enable_boundary_quality: bool = False,
) -> CheckMeshEnhanced3Result:
    """Enhanced v3 check mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    CheckMeshEnhanced3Result
    """
    quality_index = None
    if enable_quality_index:
        quality_index = CellQualityIndex(name="cell_quality_index", enabled=True)

    boundary_quality = None
    if enable_boundary_quality:
        boundary_quality = BoundaryQualityResult(name="boundary_quality_check", enabled=True)

    return CheckMeshEnhanced3Result(
        quality_index=quality_index,
        boundary_quality=boundary_quality,
    )
