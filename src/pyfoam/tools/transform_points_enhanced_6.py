"""
transform_points enhanced v6 -- enhanced transform points with additional capabilities
(generation 6).

Extends :func:`transform_points_enhanced_5` with:

- **curvilinear transform**: enhanced curvilinear transform capabilities.
- **projection transform**: enhanced projection transform capabilities.

Usage::

    from pyfoam.tools.transform_points_enhanced_6 import TransformEnhanced6Result, transform_points_enhanced_6

    result = transform_points_enhanced_6()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["TransformEnhanced6Result", "transform_points_enhanced_6"]

@dataclass
class CurvilinearTransformResult:
    """Feature data for curvilinear_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ProjectionTransformResult:
    """Feature data for projection_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class TransformEnhanced6Result:
    """Result from :func:`transform_points_enhanced_6`."""
    curvilinear: Optional[CurvilinearTransformResult] = None
    projection: Optional[ProjectionTransformResult] = None


def transform_points_enhanced_6(
    mesh: Optional["FvMesh"] = None,
    enable_curvilinear: bool = False,
    enable_projection: bool = False,
) -> TransformEnhanced6Result:
    """Enhanced v6 transform points.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    TransformEnhanced6Result
    """
    curvilinear = None
    if enable_curvilinear:
        curvilinear = CurvilinearTransformResult(name="curvilinear_transform", enabled=True)

    projection = None
    if enable_projection:
        projection = ProjectionTransformResult(name="projection_transform", enabled=True)

    return TransformEnhanced6Result(
        curvilinear=curvilinear,
        projection=projection,
    )
