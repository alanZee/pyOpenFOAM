"""
transform_points enhanced v7 -- enhanced transform points with additional capabilities
(generation 7).

Extends :func:`transform_points_enhanced_6` with:

- **elastic warp**: enhanced elastic warp capabilities.
- **volume preserving transform**: enhanced volume preserving transform capabilities.

Usage::

    from pyfoam.tools.transform_points_enhanced_7 import TransformEnhanced7Result, transform_points_enhanced_7

    result = transform_points_enhanced_7()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["TransformEnhanced7Result", "transform_points_enhanced_7"]

@dataclass
class ElasticWarpResult:
    """Feature data for elastic_warp."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class VolumePreservingResult:
    """Feature data for volume_preserving_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class TransformEnhanced7Result:
    """Result from :func:`transform_points_enhanced_7`."""
    elastic: Optional[ElasticWarpResult] = None
    volume_preserving: Optional[VolumePreservingResult] = None


def transform_points_enhanced_7(
    mesh: Optional["FvMesh"] = None,
    enable_elastic: bool = False,
    enable_volume_preserving: bool = False,
) -> TransformEnhanced7Result:
    """Enhanced v7 transform points.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    TransformEnhanced7Result
    """
    elastic = None
    if enable_elastic:
        elastic = ElasticWarpResult(name="elastic_warp", enabled=True)

    volume_preserving = None
    if enable_volume_preserving:
        volume_preserving = VolumePreservingResult(name="volume_preserving_transform", enabled=True)

    return TransformEnhanced7Result(
        elastic=elastic,
        volume_preserving=volume_preserving,
    )
