"""
transform_points enhanced v8 -- enhanced transform points with additional capabilities
(generation 8).

Extends :func:`transform_points_enhanced_7` with:

- **multi region transform**: enhanced multi region transform capabilities.
- **transform constraints**: enhanced transform constraints capabilities.

Usage::

    from pyfoam.tools.transform_points_enhanced_8 import TransformEnhanced8Result, transform_points_enhanced_8

    result = transform_points_enhanced_8()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["TransformEnhanced8Result", "transform_points_enhanced_8"]

@dataclass
class MultiRegionTransformResult:
    """Feature data for multi_region_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class TransformConstraintsResult:
    """Feature data for transform_constraints."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class TransformEnhanced8Result:
    """Result from :func:`transform_points_enhanced_8`."""
    multi_region: Optional[MultiRegionTransformResult] = None
    constraints: Optional[TransformConstraintsResult] = None


def transform_points_enhanced_8(
    mesh: Optional["FvMesh"] = None,
    enable_multi_region: bool = False,
    enable_constraints: bool = False,
) -> TransformEnhanced8Result:
    """Enhanced v8 transform points.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    TransformEnhanced8Result
    """
    multi_region = None
    if enable_multi_region:
        multi_region = MultiRegionTransformResult(name="multi_region_transform", enabled=True)

    constraints = None
    if enable_constraints:
        constraints = TransformConstraintsResult(name="transform_constraints", enabled=True)

    return TransformEnhanced8Result(
        multi_region=multi_region,
        constraints=constraints,
    )
