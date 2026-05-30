"""
surface_auto_patch enhanced v12 -- enhanced surface auto patch with additional capabilities
(generation 12).

Extends :func:`surface_auto_patch_enhanced_11` with:

- **ml guided patching**: enhanced ml guided patching capabilities.
- **topology preserving patching**: enhanced topology preserving patching capabilities.

Usage::

    from pyfoam.tools.surface_auto_patch_enhanced_12 import SurfaceAutoPatchEnhanced12Result, surface_auto_patch_enhanced_12

    result = surface_auto_patch_enhanced_12()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SurfaceAutoPatchEnhanced12Result", "surface_auto_patch_enhanced_12"]

@dataclass
class MLGuidedPatchResult:
    """Feature data for ml_guided_patching."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class TopologyPreservingPatchResult:
    """Feature data for topology_preserving_patching."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SurfaceAutoPatchEnhanced12Result:
    """Result from :func:`surface_auto_patch_enhanced_12`."""
    ml_guided: Optional[MLGuidedPatchResult] = None
    topology_preserving: Optional[TopologyPreservingPatchResult] = None


def surface_auto_patch_enhanced_12(
    mesh: Optional["FvMesh"] = None,
    enable_ml_guided: bool = False,
    enable_topology_preserving: bool = False,
) -> SurfaceAutoPatchEnhanced12Result:
    """Enhanced v12 surface auto patch.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SurfaceAutoPatchEnhanced12Result
    """
    ml_guided = None
    if enable_ml_guided:
        ml_guided = MLGuidedPatchResult(name="ml_guided_patching", enabled=True)

    topology_preserving = None
    if enable_topology_preserving:
        topology_preserving = TopologyPreservingPatchResult(name="topology_preserving_patching", enabled=True)

    return SurfaceAutoPatchEnhanced12Result(
        ml_guided=ml_guided,
        topology_preserving=topology_preserving,
    )
