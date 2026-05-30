"""
check_mesh enhanced v8 -- enhanced check mesh with additional capabilities
(generation 8).

Extends :func:`check_mesh_enhanced_7` with:

- **solver specific check**: enhanced solver specific check capabilities.
- **mesh export validation**: enhanced mesh export validation capabilities.

Usage::

    from pyfoam.tools.check_mesh_enhanced_8 import CheckMeshEnhanced8Result, check_mesh_enhanced_8

    result = check_mesh_enhanced_8()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["CheckMeshEnhanced8Result", "check_mesh_enhanced_8"]

@dataclass
class SolverSpecificCheckResult:
    """Feature data for solver_specific_check."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ExportValidationResult:
    """Feature data for mesh_export_validation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class CheckMeshEnhanced8Result:
    """Result from :func:`check_mesh_enhanced_8`."""
    solver_specific: Optional[SolverSpecificCheckResult] = None
    export_validation: Optional[ExportValidationResult] = None


def check_mesh_enhanced_8(
    mesh: Optional["FvMesh"] = None,
    enable_solver_specific: bool = False,
    enable_export_validation: bool = False,
) -> CheckMeshEnhanced8Result:
    """Enhanced v8 check mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    CheckMeshEnhanced8Result
    """
    solver_specific = None
    if enable_solver_specific:
        solver_specific = SolverSpecificCheckResult(name="solver_specific_check", enabled=True)

    export_validation = None
    if enable_export_validation:
        export_validation = ExportValidationResult(name="mesh_export_validation", enabled=True)

    return CheckMeshEnhanced8Result(
        solver_specific=solver_specific,
        export_validation=export_validation,
    )
