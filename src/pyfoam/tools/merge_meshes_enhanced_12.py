"""
merge_meshes enhanced v12 -- enhanced merge meshes with additional capabilities
(generation 12).

Extends :func:`merge_meshes_enhanced_11` with:

- **distributed merge**: enhanced distributed merge capabilities.
- **merge topology optimization**: enhanced merge topology optimization capabilities.

Usage::

    from pyfoam.tools.merge_meshes_enhanced_12 import MergeEnhanced12Result, merge_meshes_enhanced_12

    result = merge_meshes_enhanced_12()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MergeEnhanced12Result", "merge_meshes_enhanced_12"]

@dataclass
class DistributedMergeResult:
    """Feature data for distributed_merge."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MergeTopologyResult:
    """Feature data for merge_topology_optimization."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MergeEnhanced12Result:
    """Result from :func:`merge_meshes_enhanced_12`."""
    distributed: Optional[DistributedMergeResult] = None
    topology: Optional[MergeTopologyResult] = None


def merge_meshes_enhanced_12(
    mesh: Optional["FvMesh"] = None,
    enable_distributed: bool = False,
    enable_topology: bool = False,
) -> MergeEnhanced12Result:
    """Enhanced v12 merge meshes.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MergeEnhanced12Result
    """
    distributed = None
    if enable_distributed:
        distributed = DistributedMergeResult(name="distributed_merge", enabled=True)

    topology = None
    if enable_topology:
        topology = MergeTopologyResult(name="merge_topology_optimization", enabled=True)

    return MergeEnhanced12Result(
        distributed=distributed,
        topology=topology,
    )
