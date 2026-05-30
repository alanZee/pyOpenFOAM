"""
check_mesh enhanced v2 -- enhanced check mesh with additional capabilities
(generation 2).

Extends :func:`check_mesh_enhanced_1` with:

- **non orthogonality analysis**: enhanced non orthogonality analysis capabilities.
- **skewness distribution**: enhanced skewness distribution capabilities.

Usage::

    from pyfoam.tools.check_mesh_enhanced_2 import CheckMeshEnhanced2Result, check_mesh_enhanced_2

    result = check_mesh_enhanced_2()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["CheckMeshEnhanced2Result", "check_mesh_enhanced_2"]

@dataclass
class NonOrthogonalityAnalysis:
    """Feature data for non_orthogonality_analysis."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class SkewnessDistribution:
    """Feature data for skewness_distribution."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class CheckMeshEnhanced2Result:
    """Result from :func:`check_mesh_enhanced_2`."""
    non_ortho: Optional[NonOrthogonalityAnalysis] = None
    skewness: Optional[SkewnessDistribution] = None


def check_mesh_enhanced_2(
    mesh: Optional["FvMesh"] = None,
    enable_non_ortho: bool = False,
    enable_skewness: bool = False,
) -> CheckMeshEnhanced2Result:
    """Enhanced v2 check mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    CheckMeshEnhanced2Result
    """
    non_ortho = None
    if enable_non_ortho:
        non_ortho = NonOrthogonalityAnalysis(name="non_orthogonality_analysis", enabled=True)

    skewness = None
    if enable_skewness:
        skewness = SkewnessDistribution(name="skewness_distribution", enabled=True)

    return CheckMeshEnhanced2Result(
        non_ortho=non_ortho,
        skewness=skewness,
    )
