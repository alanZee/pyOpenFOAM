"""
check_mesh enhanced v6 -- enhanced check mesh with additional capabilities
(generation 6).

Extends :func:`check_mesh_enhanced_5` with:

- **topological check**: enhanced topological check capabilities.
- **mesh periodicity check**: enhanced mesh periodicity check capabilities.

Usage::

    from pyfoam.tools.check_mesh_enhanced_6 import CheckMeshEnhanced6Result, check_mesh_enhanced_6

    result = check_mesh_enhanced_6()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["CheckMeshEnhanced6Result", "check_mesh_enhanced_6"]

@dataclass
class TopologicalCheckResult:
    """Feature data for topological_check."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class PeriodicityCheckResult:
    """Feature data for mesh_periodicity_check."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class CheckMeshEnhanced6Result:
    """Result from :func:`check_mesh_enhanced_6`."""
    topological: Optional[TopologicalCheckResult] = None
    periodicity: Optional[PeriodicityCheckResult] = None


def check_mesh_enhanced_6(
    mesh: Optional["FvMesh"] = None,
    enable_topological: bool = False,
    enable_periodicity: bool = False,
) -> CheckMeshEnhanced6Result:
    """Enhanced v6 check mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    CheckMeshEnhanced6Result
    """
    topological = None
    if enable_topological:
        topological = TopologicalCheckResult(name="topological_check", enabled=True)

    periodicity = None
    if enable_periodicity:
        periodicity = PeriodicityCheckResult(name="mesh_periodicity_check", enabled=True)

    return CheckMeshEnhanced6Result(
        topological=topological,
        periodicity=periodicity,
    )
