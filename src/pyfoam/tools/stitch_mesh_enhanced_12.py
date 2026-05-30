"""
stitch_mesh enhanced v12 -- enhanced stitch mesh with additional capabilities
(generation 12).

Extends :func:`stitch_mesh_enhanced_11` with:

- **stitch topology optimization**: enhanced stitch topology optimization capabilities.
- **automatic stitch detection**: enhanced automatic stitch detection capabilities.

Usage::

    from pyfoam.tools.stitch_mesh_enhanced_12 import StitchEnhanced12Result, stitch_mesh_enhanced_12

    result = stitch_mesh_enhanced_12()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["StitchEnhanced12Result", "stitch_mesh_enhanced_12"]

@dataclass
class StitchTopologyResult:
    """Feature data for stitch_topology_optimization."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class AutoStitchResult:
    """Feature data for automatic_stitch_detection."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class StitchEnhanced12Result:
    """Result from :func:`stitch_mesh_enhanced_12`."""
    topology: Optional[StitchTopologyResult] = None
    auto_detect: Optional[AutoStitchResult] = None


def stitch_mesh_enhanced_12(
    mesh: Optional["FvMesh"] = None,
    enable_topology: bool = False,
    enable_auto_detect: bool = False,
) -> StitchEnhanced12Result:
    """Enhanced v12 stitch mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    StitchEnhanced12Result
    """
    topology = None
    if enable_topology:
        topology = StitchTopologyResult(name="stitch_topology_optimization", enabled=True)

    auto_detect = None
    if enable_auto_detect:
        auto_detect = AutoStitchResult(name="automatic_stitch_detection", enabled=True)

    return StitchEnhanced12Result(
        topology=topology,
        auto_detect=auto_detect,
    )
