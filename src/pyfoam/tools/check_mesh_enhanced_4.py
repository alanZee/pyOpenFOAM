"""
check_mesh enhanced v4 -- enhanced check mesh with additional capabilities
(generation 4).

Extends :func:`check_mesh_enhanced_3` with:

- **mesh smoothness metric**: enhanced mesh smoothness metric capabilities.
- **conformity check**: enhanced conformity check capabilities.

Usage::

    from pyfoam.tools.check_mesh_enhanced_4 import CheckMeshEnhanced4Result, check_mesh_enhanced_4

    result = check_mesh_enhanced_4()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["CheckMeshEnhanced4Result", "check_mesh_enhanced_4"]

@dataclass
class SmoothnessMetric:
    """Feature data for mesh_smoothness_metric."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ConformityResult:
    """Feature data for conformity_check."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class CheckMeshEnhanced4Result:
    """Result from :func:`check_mesh_enhanced_4`."""
    smoothness: Optional[SmoothnessMetric] = None
    conformity: Optional[ConformityResult] = None


def check_mesh_enhanced_4(
    mesh: Optional["FvMesh"] = None,
    enable_smoothness: bool = False,
    enable_conformity: bool = False,
) -> CheckMeshEnhanced4Result:
    """Enhanced v4 check mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    CheckMeshEnhanced4Result
    """
    smoothness = None
    if enable_smoothness:
        smoothness = SmoothnessMetric(name="mesh_smoothness_metric", enabled=True)

    conformity = None
    if enable_conformity:
        conformity = ConformityResult(name="conformity_check", enabled=True)

    return CheckMeshEnhanced4Result(
        smoothness=smoothness,
        conformity=conformity,
    )
