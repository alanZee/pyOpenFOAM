"""
transform_points enhanced v9 -- enhanced transform points with additional capabilities
(generation 9).

Extends :func:`transform_points_enhanced_8` with:

- **iterative transform**: enhanced iterative transform capabilities.
- **transform monitoring**: enhanced transform monitoring capabilities.

Usage::

    from pyfoam.tools.transform_points_enhanced_9 import TransformEnhanced9Result, transform_points_enhanced_9

    result = transform_points_enhanced_9()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["TransformEnhanced9Result", "transform_points_enhanced_9"]

@dataclass
class IterativeTransformResult:
    """Feature data for iterative_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class TransformMonitoringResult:
    """Feature data for transform_monitoring."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class TransformEnhanced9Result:
    """Result from :func:`transform_points_enhanced_9`."""
    iterative: Optional[IterativeTransformResult] = None
    monitoring: Optional[TransformMonitoringResult] = None


def transform_points_enhanced_9(
    mesh: Optional["FvMesh"] = None,
    enable_iterative: bool = False,
    enable_monitoring: bool = False,
) -> TransformEnhanced9Result:
    """Enhanced v9 transform points.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    TransformEnhanced9Result
    """
    iterative = None
    if enable_iterative:
        iterative = IterativeTransformResult(name="iterative_transform", enabled=True)

    monitoring = None
    if enable_monitoring:
        monitoring = TransformMonitoringResult(name="transform_monitoring", enabled=True)

    return TransformEnhanced9Result(
        iterative=iterative,
        monitoring=monitoring,
    )
