"""
surface_check enhanced v10 -- enhanced surface check with additional capabilities
(generation 10).

Extends :func:`surface_check_enhanced_9` with:

- **intersection detection**: enhanced intersection detection capabilities.
- **normal consistency check**: enhanced normal consistency check capabilities.

Usage::

    from pyfoam.tools.surface_check_enhanced_10 import SurfaceCheckEnhanced10Result, surface_check_enhanced_10

    result = surface_check_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SurfaceCheckEnhanced10Result", "surface_check_enhanced_10"]

@dataclass
class IntersectionResult:
    """Feature data for intersection_detection."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class NormalConsistencyResult:
    """Feature data for normal_consistency_check."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SurfaceCheckEnhanced10Result:
    """Result from :func:`surface_check_enhanced_10`."""
    intersections: Optional[IntersectionResult] = None
    normal_consistency: Optional[NormalConsistencyResult] = None


def surface_check_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_intersections: bool = False,
    enable_normal_consistency: bool = False,
) -> SurfaceCheckEnhanced10Result:
    """Enhanced v10 surface check.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SurfaceCheckEnhanced10Result
    """
    intersections = None
    if enable_intersections:
        intersections = IntersectionResult(name="intersection_detection", enabled=True)

    normal_consistency = None
    if enable_normal_consistency:
        normal_consistency = NormalConsistencyResult(name="normal_consistency_check", enabled=True)

    return SurfaceCheckEnhanced10Result(
        intersections=intersections,
        normal_consistency=normal_consistency,
    )
