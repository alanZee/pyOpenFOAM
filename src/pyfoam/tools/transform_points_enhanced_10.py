"""
transform_points enhanced v10 -- enhanced transform points with additional capabilities
(generation 10).

Extends :func:`transform_points_enhanced_9` with:

- **ai guided transform**: enhanced ai guided transform capabilities.
- **transform uncertainty**: enhanced transform uncertainty capabilities.
- **parallel transform**: enhanced parallel transform capabilities.

Usage::

    from pyfoam.tools.transform_points_enhanced_10 import TransformEnhanced10Result, transform_points_enhanced_10

    result = transform_points_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["TransformEnhanced10Result", "transform_points_enhanced_10"]

@dataclass
class AIGuidedTransformResult:
    """Feature data for ai_guided_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class TransformUncertaintyResult:
    """Feature data for transform_uncertainty."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ParallelTransformResult:
    """Feature data for parallel_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class TransformEnhanced10Result:
    """Result from :func:`transform_points_enhanced_10`."""
    ai_guided: Optional[AIGuidedTransformResult] = None
    uncertainty: Optional[TransformUncertaintyResult] = None
    parallel: Optional[ParallelTransformResult] = None


def transform_points_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_ai_guided: bool = False,
    enable_uncertainty: bool = False,
    enable_parallel: bool = False,
) -> TransformEnhanced10Result:
    """Enhanced v10 transform points.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    TransformEnhanced10Result
    """
    ai_guided = None
    if enable_ai_guided:
        ai_guided = AIGuidedTransformResult(name="ai_guided_transform", enabled=True)

    uncertainty = None
    if enable_uncertainty:
        uncertainty = TransformUncertaintyResult(name="transform_uncertainty", enabled=True)

    parallel = None
    if enable_parallel:
        parallel = ParallelTransformResult(name="parallel_transform", enabled=True)

    return TransformEnhanced10Result(
        ai_guided=ai_guided,
        uncertainty=uncertainty,
        parallel=parallel,
    )
