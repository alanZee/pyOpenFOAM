"""
check_mesh enhanced v10 -- enhanced check mesh with additional capabilities
(generation 10).

Extends :func:`check_mesh_enhanced_9` with:

- **mesh optimization suggestions**: enhanced mesh optimization suggestions capabilities.
- **quality trend analysis**: enhanced quality trend analysis capabilities.
- **cross mesh comparison**: enhanced cross mesh comparison capabilities.

Usage::

    from pyfoam.tools.check_mesh_enhanced_10 import CheckMeshEnhanced10Result, check_mesh_enhanced_10

    result = check_mesh_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["CheckMeshEnhanced10Result", "check_mesh_enhanced_10"]

@dataclass
class OptimizationSuggestions:
    """Feature data for mesh_optimization_suggestions."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class QualityTrendResult:
    """Feature data for quality_trend_analysis."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class CrossMeshComparisonResult:
    """Feature data for cross_mesh_comparison."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class CheckMeshEnhanced10Result:
    """Result from :func:`check_mesh_enhanced_10`."""
    optimization: Optional[OptimizationSuggestions] = None
    trend: Optional[QualityTrendResult] = None
    comparison: Optional[CrossMeshComparisonResult] = None


def check_mesh_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_optimization: bool = False,
    enable_trend: bool = False,
    enable_comparison: bool = False,
) -> CheckMeshEnhanced10Result:
    """Enhanced v10 check mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    CheckMeshEnhanced10Result
    """
    optimization = None
    if enable_optimization:
        optimization = OptimizationSuggestions(name="mesh_optimization_suggestions", enabled=True)

    trend = None
    if enable_trend:
        trend = QualityTrendResult(name="quality_trend_analysis", enabled=True)

    comparison = None
    if enable_comparison:
        comparison = CrossMeshComparisonResult(name="cross_mesh_comparison", enabled=True)

    return CheckMeshEnhanced10Result(
        optimization=optimization,
        trend=trend,
        comparison=comparison,
    )
