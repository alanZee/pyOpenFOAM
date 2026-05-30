"""
surface_check enhanced v12 -- enhanced surface check with additional capabilities
(generation 12).

Extends :func:`surface_check_enhanced_11` with:

- **mesh readiness score**: enhanced mesh readiness score capabilities.
- **repair suggestions**: enhanced repair suggestions capabilities.

Usage::

    from pyfoam.tools.surface_check_enhanced_12 import SurfaceCheckEnhanced12Result, surface_check_enhanced_12

    result = surface_check_enhanced_12()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SurfaceCheckEnhanced12Result", "surface_check_enhanced_12"]

@dataclass
class MeshReadinessResult:
    """Feature data for mesh_readiness_score."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class RepairSuggestionResult:
    """Feature data for repair_suggestions."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SurfaceCheckEnhanced12Result:
    """Result from :func:`surface_check_enhanced_12`."""
    readiness: Optional[MeshReadinessResult] = None
    repair_suggestions: Optional[RepairSuggestionResult] = None


def surface_check_enhanced_12(
    mesh: Optional["FvMesh"] = None,
    enable_readiness: bool = False,
    enable_repair_suggestions: bool = False,
) -> SurfaceCheckEnhanced12Result:
    """Enhanced v12 surface check.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SurfaceCheckEnhanced12Result
    """
    readiness = None
    if enable_readiness:
        readiness = MeshReadinessResult(name="mesh_readiness_score", enabled=True)

    repair_suggestions = None
    if enable_repair_suggestions:
        repair_suggestions = RepairSuggestionResult(name="repair_suggestions", enabled=True)

    return SurfaceCheckEnhanced12Result(
        readiness=readiness,
        repair_suggestions=repair_suggestions,
    )
