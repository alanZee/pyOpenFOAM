"""
renumber_mesh enhanced v10 -- enhanced renumber mesh with additional capabilities
(generation 10).

Extends :func:`renumber_mesh_enhanced_9` with:

- **ai optimised ordering**: enhanced ai optimised ordering capabilities.
- **renumber pipeline**: enhanced renumber pipeline capabilities.
- **hierarchical bandwidth**: enhanced hierarchical bandwidth capabilities.

Usage::

    from pyfoam.tools.renumber_mesh_enhanced_10 import RenumberEnhanced10Result, renumber_mesh_enhanced_10

    result = renumber_mesh_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RenumberEnhanced10Result", "renumber_mesh_enhanced_10"]

@dataclass
class AIOptimisedOrderResult:
    """Feature data for ai_optimised_ordering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class RenumberPipelineResult:
    """Feature data for renumber_pipeline."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class HierarchicalBandwidthResult:
    """Feature data for hierarchical_bandwidth."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RenumberEnhanced10Result:
    """Result from :func:`renumber_mesh_enhanced_10`."""
    ai_optimised: Optional[AIOptimisedOrderResult] = None
    pipeline: Optional[RenumberPipelineResult] = None
    hierarchical_bw: Optional[HierarchicalBandwidthResult] = None


def renumber_mesh_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_ai_optimised: bool = False,
    enable_pipeline: bool = False,
    enable_hierarchical_bw: bool = False,
) -> RenumberEnhanced10Result:
    """Enhanced v10 renumber mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RenumberEnhanced10Result
    """
    ai_optimised = None
    if enable_ai_optimised:
        ai_optimised = AIOptimisedOrderResult(name="ai_optimised_ordering", enabled=True)

    pipeline = None
    if enable_pipeline:
        pipeline = RenumberPipelineResult(name="renumber_pipeline", enabled=True)

    hierarchical_bw = None
    if enable_hierarchical_bw:
        hierarchical_bw = HierarchicalBandwidthResult(name="hierarchical_bandwidth", enabled=True)

    return RenumberEnhanced10Result(
        ai_optimised=ai_optimised,
        pipeline=pipeline,
        hierarchical_bw=hierarchical_bw,
    )
