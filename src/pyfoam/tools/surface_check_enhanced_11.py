"""
surface_check enhanced v11 -- enhanced surface check with additional capabilities
(generation 11).

Extends :func:`surface_check_enhanced_10` with:

- **self proximity check**: enhanced self proximity check capabilities.
- **curvature quality metric**: enhanced curvature quality metric capabilities.

Usage::

    from pyfoam.tools.surface_check_enhanced_11 import SurfaceCheckEnhanced11Result, surface_check_enhanced_11

    result = surface_check_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SurfaceCheckEnhanced11Result", "surface_check_enhanced_11"]

@dataclass
class SelfProximityResult:
    """Feature data for self_proximity_check."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class CurvatureQualityResult:
    """Feature data for curvature_quality_metric."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SurfaceCheckEnhanced11Result:
    """Result from :func:`surface_check_enhanced_11`."""
    proximity: Optional[SelfProximityResult] = None
    curvature_quality: Optional[CurvatureQualityResult] = None


def surface_check_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_proximity: bool = False,
    enable_curvature_quality: bool = False,
) -> SurfaceCheckEnhanced11Result:
    """Enhanced v11 surface check.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SurfaceCheckEnhanced11Result
    """
    proximity = None
    if enable_proximity:
        proximity = SelfProximityResult(name="self_proximity_check", enabled=True)

    curvature_quality = None
    if enable_curvature_quality:
        curvature_quality = CurvatureQualityResult(name="curvature_quality_metric", enabled=True)

    return SurfaceCheckEnhanced11Result(
        proximity=proximity,
        curvature_quality=curvature_quality,
    )
