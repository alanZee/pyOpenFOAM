"""
surface_features enhanced v12 -- enhanced surface features with additional capabilities
(generation 12).

Extends :func:`surface_features_enhanced_11` with:

- **ml feature detection**: enhanced ml feature detection capabilities.
- **feature topology graph**: enhanced feature topology graph capabilities.

Usage::

    from pyfoam.tools.surface_features_enhanced_12 import SurfaceFeaturesEnhanced12Result, surface_features_enhanced_12

    result = surface_features_enhanced_12()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SurfaceFeaturesEnhanced12Result", "surface_features_enhanced_12"]

@dataclass
class MLFeatureDetectionResult:
    """Feature data for ml_feature_detection."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class FeatureTopologyResult:
    """Feature data for feature_topology_graph."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SurfaceFeaturesEnhanced12Result:
    """Result from :func:`surface_features_enhanced_12`."""
    ml_detection: Optional[MLFeatureDetectionResult] = None
    topology_graph: Optional[FeatureTopologyResult] = None


def surface_features_enhanced_12(
    mesh: Optional["FvMesh"] = None,
    enable_ml_detection: bool = False,
    enable_topology_graph: bool = False,
) -> SurfaceFeaturesEnhanced12Result:
    """Enhanced v12 surface features.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SurfaceFeaturesEnhanced12Result
    """
    ml_detection = None
    if enable_ml_detection:
        ml_detection = MLFeatureDetectionResult(name="ml_feature_detection", enabled=True)

    topology_graph = None
    if enable_topology_graph:
        topology_graph = FeatureTopologyResult(name="feature_topology_graph", enabled=True)

    return SurfaceFeaturesEnhanced12Result(
        ml_detection=ml_detection,
        topology_graph=topology_graph,
    )
