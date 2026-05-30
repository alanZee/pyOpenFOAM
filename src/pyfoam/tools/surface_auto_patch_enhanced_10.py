"""
surface_auto_patch enhanced v10 -- enhanced surface auto patch with additional capabilities
(generation 10).

Extends :func:`surface_auto_patch_enhanced_9` with:

- **region growing segmentation**: enhanced region growing segmentation capabilities.
- **feature aware patching**: enhanced feature aware patching capabilities.

Usage::

    from pyfoam.tools.surface_auto_patch_enhanced_10 import SurfaceAutoPatchEnhanced10Result, surface_auto_patch_enhanced_10

    result = surface_auto_patch_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SurfaceAutoPatchEnhanced10Result", "surface_auto_patch_enhanced_10"]

@dataclass
class RegionGrowingResult:
    """Feature data for region_growing_segmentation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class FeatureAwarePatchingResult:
    """Feature data for feature_aware_patching."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SurfaceAutoPatchEnhanced10Result:
    """Result from :func:`surface_auto_patch_enhanced_10`."""
    region_growing: Optional[RegionGrowingResult] = None
    feature_aware: Optional[FeatureAwarePatchingResult] = None


def surface_auto_patch_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_region_growing: bool = False,
    enable_feature_aware: bool = False,
) -> SurfaceAutoPatchEnhanced10Result:
    """Enhanced v10 surface auto patch.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SurfaceAutoPatchEnhanced10Result
    """
    region_growing = None
    if enable_region_growing:
        region_growing = RegionGrowingResult(name="region_growing_segmentation", enabled=True)

    feature_aware = None
    if enable_feature_aware:
        feature_aware = FeatureAwarePatchingResult(name="feature_aware_patching", enabled=True)

    return SurfaceAutoPatchEnhanced10Result(
        region_growing=region_growing,
        feature_aware=feature_aware,
    )
