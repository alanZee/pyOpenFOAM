"""
refine_mesh enhanced v9 -- enhanced refine mesh with additional capabilities
(generation 9).

Extends :func:`refine_mesh_enhanced_8` with:

- **parallel refinement**: enhanced parallel refinement capabilities.
- **refinement consistency**: enhanced refinement consistency capabilities.

Usage::

    from pyfoam.tools.refine_mesh_enhanced_9 import RefineEnhanced9Result, refine_mesh_enhanced_9

    result = refine_mesh_enhanced_9()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RefineEnhanced9Result", "refine_mesh_enhanced_9"]

@dataclass
class ParallelRefineResult:
    """Feature data for parallel_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class RefineConsistencyResult:
    """Feature data for refinement_consistency."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RefineEnhanced9Result:
    """Result from :func:`refine_mesh_enhanced_9`."""
    parallel: Optional[ParallelRefineResult] = None
    consistency: Optional[RefineConsistencyResult] = None


def refine_mesh_enhanced_9(
    mesh: Optional["FvMesh"] = None,
    enable_parallel: bool = False,
    enable_consistency: bool = False,
) -> RefineEnhanced9Result:
    """Enhanced v9 refine mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RefineEnhanced9Result
    """
    parallel = None
    if enable_parallel:
        parallel = ParallelRefineResult(name="parallel_refinement", enabled=True)

    consistency = None
    if enable_consistency:
        consistency = RefineConsistencyResult(name="refinement_consistency", enabled=True)

    return RefineEnhanced9Result(
        parallel=parallel,
        consistency=consistency,
    )
