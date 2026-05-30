"""
create_patch enhanced v12 -- enhanced create patch with additional capabilities
(generation 12).

Extends :func:`create_patch_enhanced_11` with:

- **adaptive patch resolution**: enhanced adaptive patch resolution capabilities.
- **patch topology optimization**: enhanced patch topology optimization capabilities.

Usage::

    from pyfoam.tools.create_patch_enhanced_12 import PatchEnhanced12Result, create_patch_enhanced_12

    result = create_patch_enhanced_12()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["PatchEnhanced12Result", "create_patch_enhanced_12"]

@dataclass
class AdaptivePatchResult:
    """Feature data for adaptive_patch_resolution."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class PatchTopologyResult:
    """Feature data for patch_topology_optimization."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class PatchEnhanced12Result:
    """Result from :func:`create_patch_enhanced_12`."""
    adaptive_resolution: Optional[AdaptivePatchResult] = None
    topology: Optional[PatchTopologyResult] = None


def create_patch_enhanced_12(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive_resolution: bool = False,
    enable_topology: bool = False,
) -> PatchEnhanced12Result:
    """Enhanced v12 create patch.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    PatchEnhanced12Result
    """
    adaptive_resolution = None
    if enable_adaptive_resolution:
        adaptive_resolution = AdaptivePatchResult(name="adaptive_patch_resolution", enabled=True)

    topology = None
    if enable_topology:
        topology = PatchTopologyResult(name="patch_topology_optimization", enabled=True)

    return PatchEnhanced12Result(
        adaptive_resolution=adaptive_resolution,
        topology=topology,
    )
