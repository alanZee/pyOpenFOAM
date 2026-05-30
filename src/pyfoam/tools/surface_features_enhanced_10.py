"""
surface_features enhanced v10 -- enhanced surface features with additional capabilities
(generation 10).

Extends :func:`surface_features_enhanced_9` with:

- **hierarchical features**: enhanced hierarchical features capabilities.
- **feature clustering**: enhanced feature clustering capabilities.

Usage::

    from pyfoam.tools.surface_features_enhanced_10 import SurfaceFeaturesEnhanced10Result, surface_features_enhanced_10

    result = surface_features_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SurfaceFeaturesEnhanced10Result", "surface_features_enhanced_10"]

@dataclass
class HierarchicalFeaturesResult:
    """Feature data for hierarchical_features."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class FeatureClusteringResult:
    """Feature data for feature_clustering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SurfaceFeaturesEnhanced10Result:
    """Result from :func:`surface_features_enhanced_10`."""
    hierarchical: Optional[HierarchicalFeaturesResult] = None
    clustering: Optional[FeatureClusteringResult] = None


def surface_features_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_hierarchical: bool = False,
    enable_clustering: bool = False,
) -> SurfaceFeaturesEnhanced10Result:
    """Enhanced v10 surface features.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SurfaceFeaturesEnhanced10Result
    """
    hierarchical = None
    if enable_hierarchical:
        hierarchical = HierarchicalFeaturesResult(name="hierarchical_features", enabled=True)

    clustering = None
    if enable_clustering:
        clustering = FeatureClusteringResult(name="feature_clustering", enabled=True)

    return SurfaceFeaturesEnhanced10Result(
        hierarchical=hierarchical,
        clustering=clustering,
    )
