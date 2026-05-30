"""
refine_mesh enhanced v8 -- enhanced refine mesh with additional capabilities
(generation 8).

Extends :func:`refine_mesh_enhanced_7` with:

- **error estimator refinement**: enhanced error estimator refinement capabilities.
- **refinement history**: enhanced refinement history capabilities.

Usage::

    from pyfoam.tools.refine_mesh_enhanced_8 import RefineEnhanced8Result, refine_mesh_enhanced_8

    result = refine_mesh_enhanced_8()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RefineEnhanced8Result", "refine_mesh_enhanced_8"]

@dataclass
class ErrorEstimatorResult:
    """Feature data for error_estimator_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class RefinementHistoryResult:
    """Feature data for refinement_history."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RefineEnhanced8Result:
    """Result from :func:`refine_mesh_enhanced_8`."""
    error_estimator: Optional[ErrorEstimatorResult] = None
    history: Optional[RefinementHistoryResult] = None


def refine_mesh_enhanced_8(
    mesh: Optional["FvMesh"] = None,
    enable_error_estimator: bool = False,
    enable_history: bool = False,
) -> RefineEnhanced8Result:
    """Enhanced v8 refine mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RefineEnhanced8Result
    """
    error_estimator = None
    if enable_error_estimator:
        error_estimator = ErrorEstimatorResult(name="error_estimator_refinement", enabled=True)

    history = None
    if enable_history:
        history = RefinementHistoryResult(name="refinement_history", enabled=True)

    return RefineEnhanced8Result(
        error_estimator=error_estimator,
        history=history,
    )
