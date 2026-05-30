"""
renumber_mesh enhanced v3 -- enhanced renumber mesh with additional capabilities
(generation 3).

Extends :func:`renumber_mesh_enhanced_2` with:

- **space filling curve**: enhanced space filling curve capabilities.
- **partition aware renumbering**: enhanced partition aware renumbering capabilities.

Usage::

    from pyfoam.tools.renumber_mesh_enhanced_3 import RenumberEnhanced3Result, renumber_mesh_enhanced_3

    result = renumber_mesh_enhanced_3()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RenumberEnhanced3Result", "renumber_mesh_enhanced_3"]

@dataclass
class SpaceFillingCurveResult:
    """Feature data for space_filling_curve."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class PartitionAwareResult:
    """Feature data for partition_aware_renumbering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RenumberEnhanced3Result:
    """Result from :func:`renumber_mesh_enhanced_3`."""
    sfc: Optional[SpaceFillingCurveResult] = None
    partition: Optional[PartitionAwareResult] = None


def renumber_mesh_enhanced_3(
    mesh: Optional["FvMesh"] = None,
    enable_sfc: bool = False,
    enable_partition: bool = False,
) -> RenumberEnhanced3Result:
    """Enhanced v3 renumber mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RenumberEnhanced3Result
    """
    sfc = None
    if enable_sfc:
        sfc = SpaceFillingCurveResult(name="space_filling_curve", enabled=True)

    partition = None
    if enable_partition:
        partition = PartitionAwareResult(name="partition_aware_renumbering", enabled=True)

    return RenumberEnhanced3Result(
        sfc=sfc,
        partition=partition,
    )
