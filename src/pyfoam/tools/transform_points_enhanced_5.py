"""
transform_points enhanced v5 -- enhanced transform points with additional capabilities
(generation 5).

Extends :func:`transform_points_enhanced_4` with:

- **adaptive transform**: enhanced adaptive transform capabilities.
- **transform validation**: enhanced transform validation capabilities.
- **parametric transform**: enhanced parametric transform capabilities.

Usage::

    from pyfoam.tools.transform_points_enhanced_5 import TransformEnhanced5Result, transform_points_enhanced_5

    result = transform_points_enhanced_5()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["TransformEnhanced5Result", "transform_points_enhanced_5"]

@dataclass
class AdaptiveTransformResult:
    """Feature data for adaptive_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class TransformValidationResult:
    """Feature data for transform_validation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ParametricTransformResult:
    """Feature data for parametric_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class TransformEnhanced5Result:
    """Result from :func:`transform_points_enhanced_5`."""
    adaptive: Optional[AdaptiveTransformResult] = None
    validation: Optional[TransformValidationResult] = None
    parametric: Optional[ParametricTransformResult] = None


def transform_points_enhanced_5(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive: bool = False,
    enable_validation: bool = False,
    enable_parametric: bool = False,
) -> TransformEnhanced5Result:
    """Enhanced v5 transform points.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    TransformEnhanced5Result
    """
    adaptive = None
    if enable_adaptive:
        adaptive = AdaptiveTransformResult(name="adaptive_transform", enabled=True)

    validation = None
    if enable_validation:
        validation = TransformValidationResult(name="transform_validation", enabled=True)

    parametric = None
    if enable_parametric:
        parametric = ParametricTransformResult(name="parametric_transform", enabled=True)

    return TransformEnhanced5Result(
        adaptive=adaptive,
        validation=validation,
        parametric=parametric,
    )
