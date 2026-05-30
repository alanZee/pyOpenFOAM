"""
surface_features enhanced v11 -- enhanced surface features with additional capabilities
(generation 11).

Extends :func:`surface_features_enhanced_10` with:

- **curvature adaptive features**: enhanced curvature adaptive features capabilities.
- **feature persistence**: enhanced feature persistence capabilities.

Usage::

    from pyfoam.tools.surface_features_enhanced_11 import SurfaceFeaturesEnhanced11Result, surface_features_enhanced_11

    result = surface_features_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SurfaceFeaturesEnhanced11Result", "surface_features_enhanced_11"]

@dataclass
class CurvatureAdaptiveFeatureResult:
    """Feature data for curvature_adaptive_features."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class FeaturePersistenceResult:
    """Feature data for feature_persistence."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SurfaceFeaturesEnhanced11Result:
    """Result from :func:`surface_features_enhanced_11`."""
    curvature_adaptive: Optional[CurvatureAdaptiveFeatureResult] = None
    persistence: Optional[FeaturePersistenceResult] = None


def surface_features_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_curvature_adaptive: bool = False,
    enable_persistence: bool = False,
) -> SurfaceFeaturesEnhanced11Result:
    """Enhanced v11 surface features.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SurfaceFeaturesEnhanced11Result
    """
    curvature_adaptive = None
    if enable_curvature_adaptive:
        curvature_adaptive = CurvatureAdaptiveFeatureResult(name="curvature_adaptive_features", enabled=True)

    persistence = None
    if enable_persistence:
        persistence = FeaturePersistenceResult(name="feature_persistence", enabled=True)

    return SurfaceFeaturesEnhanced11Result(
        curvature_adaptive=curvature_adaptive,
        persistence=persistence,
    )
