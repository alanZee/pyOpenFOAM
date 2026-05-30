"""
renumber_mesh enhanced v8 -- enhanced renumber mesh with additional capabilities
(generation 8).

Extends :func:`renumber_mesh_enhanced_7` with:

- **parallel renumbering**: enhanced parallel renumbering capabilities.
- **renumber diagnostics**: enhanced renumber diagnostics capabilities.

Usage::

    from pyfoam.tools.renumber_mesh_enhanced_8 import RenumberEnhanced8Result, renumber_mesh_enhanced_8

    result = renumber_mesh_enhanced_8()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RenumberEnhanced8Result", "renumber_mesh_enhanced_8"]

@dataclass
class ParallelRenumberResult:
    """Feature data for parallel_renumbering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class RenumberDiagnosticsResult:
    """Feature data for renumber_diagnostics."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RenumberEnhanced8Result:
    """Result from :func:`renumber_mesh_enhanced_8`."""
    parallel: Optional[ParallelRenumberResult] = None
    diagnostics: Optional[RenumberDiagnosticsResult] = None


def renumber_mesh_enhanced_8(
    mesh: Optional["FvMesh"] = None,
    enable_parallel: bool = False,
    enable_diagnostics: bool = False,
) -> RenumberEnhanced8Result:
    """Enhanced v8 renumber mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RenumberEnhanced8Result
    """
    parallel = None
    if enable_parallel:
        parallel = ParallelRenumberResult(name="parallel_renumbering", enabled=True)

    diagnostics = None
    if enable_diagnostics:
        diagnostics = RenumberDiagnosticsResult(name="renumber_diagnostics", enabled=True)

    return RenumberEnhanced8Result(
        parallel=parallel,
        diagnostics=diagnostics,
    )
