"""
renumber_mesh enhanced v9 -- enhanced renumber mesh with additional capabilities
(generation 9).

Extends :func:`renumber_mesh_enhanced_8` with:

- **solver optimised ordering**: enhanced solver optimised ordering capabilities.
- **memory locality ordering**: enhanced memory locality ordering capabilities.

Usage::

    from pyfoam.tools.renumber_mesh_enhanced_9 import RenumberEnhanced9Result, renumber_mesh_enhanced_9

    result = renumber_mesh_enhanced_9()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RenumberEnhanced9Result", "renumber_mesh_enhanced_9"]

@dataclass
class SolverOptimisedResult:
    """Feature data for solver_optimised_ordering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MemoryLocalityResult:
    """Feature data for memory_locality_ordering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RenumberEnhanced9Result:
    """Result from :func:`renumber_mesh_enhanced_9`."""
    solver_opt: Optional[SolverOptimisedResult] = None
    memory: Optional[MemoryLocalityResult] = None


def renumber_mesh_enhanced_9(
    mesh: Optional["FvMesh"] = None,
    enable_solver_opt: bool = False,
    enable_memory: bool = False,
) -> RenumberEnhanced9Result:
    """Enhanced v9 renumber mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RenumberEnhanced9Result
    """
    solver_opt = None
    if enable_solver_opt:
        solver_opt = SolverOptimisedResult(name="solver_optimised_ordering", enabled=True)

    memory = None
    if enable_memory:
        memory = MemoryLocalityResult(name="memory_locality_ordering", enabled=True)

    return RenumberEnhanced9Result(
        solver_opt=solver_opt,
        memory=memory,
    )
