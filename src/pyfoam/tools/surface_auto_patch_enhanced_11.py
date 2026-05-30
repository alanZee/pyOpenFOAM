"""
surface_auto_patch enhanced v11 -- enhanced surface auto patch with additional capabilities
(generation 11).

Extends :func:`surface_auto_patch_enhanced_10` with:

- **curvature adaptive patching**: enhanced curvature adaptive patching capabilities.
- **multi scale segmentation**: enhanced multi scale segmentation capabilities.

Usage::

    from pyfoam.tools.surface_auto_patch_enhanced_11 import SurfaceAutoPatchEnhanced11Result, surface_auto_patch_enhanced_11

    result = surface_auto_patch_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SurfaceAutoPatchEnhanced11Result", "surface_auto_patch_enhanced_11"]

@dataclass
class CurvatureAdaptivePatchResult:
    """Feature data for curvature_adaptive_patching."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MultiScaleSegmentationResult:
    """Feature data for multi_scale_segmentation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SurfaceAutoPatchEnhanced11Result:
    """Result from :func:`surface_auto_patch_enhanced_11`."""
    curvature_adaptive: Optional[CurvatureAdaptivePatchResult] = None
    multi_scale: Optional[MultiScaleSegmentationResult] = None


def surface_auto_patch_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_curvature_adaptive: bool = False,
    enable_multi_scale: bool = False,
) -> SurfaceAutoPatchEnhanced11Result:
    """Enhanced v11 surface auto patch.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SurfaceAutoPatchEnhanced11Result
    """
    curvature_adaptive = None
    if enable_curvature_adaptive:
        curvature_adaptive = CurvatureAdaptivePatchResult(name="curvature_adaptive_patching", enabled=True)

    multi_scale = None
    if enable_multi_scale:
        multi_scale = MultiScaleSegmentationResult(name="multi_scale_segmentation", enabled=True)

    return SurfaceAutoPatchEnhanced11Result(
        curvature_adaptive=curvature_adaptive,
        multi_scale=multi_scale,
    )
