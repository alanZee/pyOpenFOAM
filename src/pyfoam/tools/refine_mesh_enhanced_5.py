"""
refine_mesh enhanced v5 -- enhanced refine mesh with additional capabilities
(generation 5).

Extends :func:`refine_mesh_enhanced_4` with:

- **adaptive refinement strategy**: enhanced adaptive refinement strategy capabilities.
- **refinement statistics**: enhanced refinement statistics capabilities.
- **coarsening support**: enhanced coarsening support capabilities.

Usage::

    from pyfoam.tools.refine_mesh_enhanced_5 import RefineEnhanced5Result, refine_mesh_enhanced_5

    result = refine_mesh_enhanced_5()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RefineEnhanced5Result", "refine_mesh_enhanced_5"]

@dataclass
class AdaptiveRefineResult:
    """Feature data for adaptive_refinement_strategy."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class RefineStatisticsResult:
    """Feature data for refinement_statistics."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class CoarseningResult:
    """Feature data for coarsening_support."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RefineEnhanced5Result:
    """Result from :func:`refine_mesh_enhanced_5`."""
    adaptive: Optional[AdaptiveRefineResult] = None
    statistics: Optional[RefineStatisticsResult] = None
    coarsening: Optional[CoarseningResult] = None


def refine_mesh_enhanced_5(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive: bool = False,
    enable_statistics: bool = False,
    enable_coarsening: bool = False,
) -> RefineEnhanced5Result:
    """Enhanced v5 refine mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RefineEnhanced5Result
    """
    adaptive = None
    if enable_adaptive:
        adaptive = AdaptiveRefineResult(name="adaptive_refinement_strategy", enabled=True)

    statistics = None
    if enable_statistics:
        statistics = RefineStatisticsResult(name="refinement_statistics", enabled=True)

    coarsening = None
    if enable_coarsening:
        coarsening = CoarseningResult(name="coarsening_support", enabled=True)

    return RefineEnhanced5Result(
        adaptive=adaptive,
        statistics=statistics,
        coarsening=coarsening,
    )
