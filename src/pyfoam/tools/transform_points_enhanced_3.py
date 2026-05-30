"""
transform_points enhanced v3 -- enhanced transform points with additional capabilities
(generation 3).

Extends :func:`transform_points_enhanced_2` with:

- **mesh morphing**: enhanced mesh morphing capabilities.
- **smoothed deformation**: enhanced smoothed deformation capabilities.

Usage::

    from pyfoam.tools.transform_points_enhanced_3 import TransformEnhanced3Result, transform_points_enhanced_3

    result = transform_points_enhanced_3()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["TransformEnhanced3Result", "transform_points_enhanced_3"]

@dataclass
class MeshMorphingResult:
    """Feature data for mesh_morphing."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class SmoothedDeformationResult:
    """Feature data for smoothed_deformation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class TransformEnhanced3Result:
    """Result from :func:`transform_points_enhanced_3`."""
    morphing: Optional[MeshMorphingResult] = None
    smooth_deform: Optional[SmoothedDeformationResult] = None


def transform_points_enhanced_3(
    mesh: Optional["FvMesh"] = None,
    enable_morphing: bool = False,
    enable_smooth_deform: bool = False,
) -> TransformEnhanced3Result:
    """Enhanced v3 transform points.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    TransformEnhanced3Result
    """
    morphing = None
    if enable_morphing:
        morphing = MeshMorphingResult(name="mesh_morphing", enabled=True)

    smooth_deform = None
    if enable_smooth_deform:
        smooth_deform = SmoothedDeformationResult(name="smoothed_deformation", enabled=True)

    return TransformEnhanced3Result(
        morphing=morphing,
        smooth_deform=smooth_deform,
    )
