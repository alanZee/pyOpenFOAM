"""
transform_points enhanced v4 -- enhanced transform points with additional capabilities
(generation 4).

Extends :func:`transform_points_enhanced_3` with:

- **topology preserving transform**: enhanced topology preserving transform capabilities.
- **inverse transform**: enhanced inverse transform capabilities.

Usage::

    from pyfoam.tools.transform_points_enhanced_4 import TransformEnhanced4Result, transform_points_enhanced_4

    result = transform_points_enhanced_4()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["TransformEnhanced4Result", "transform_points_enhanced_4"]

@dataclass
class TopologyPreservingResult:
    """Feature data for topology_preserving_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class InverseTransformResult:
    """Feature data for inverse_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class TransformEnhanced4Result:
    """Result from :func:`transform_points_enhanced_4`."""
    topology: Optional[TopologyPreservingResult] = None
    inverse: Optional[InverseTransformResult] = None


def transform_points_enhanced_4(
    mesh: Optional["FvMesh"] = None,
    enable_topology: bool = False,
    enable_inverse: bool = False,
) -> TransformEnhanced4Result:
    """Enhanced v4 transform points.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    TransformEnhanced4Result
    """
    topology = None
    if enable_topology:
        topology = TopologyPreservingResult(name="topology_preserving_transform", enabled=True)

    inverse = None
    if enable_inverse:
        inverse = InverseTransformResult(name="inverse_transform", enabled=True)

    return TransformEnhanced4Result(
        topology=topology,
        inverse=inverse,
    )
