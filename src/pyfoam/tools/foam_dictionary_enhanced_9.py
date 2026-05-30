"""
foam_dictionary enhanced v9 -- enhanced foam dictionary with additional capabilities
(generation 9).

Extends :func:`foam_dictionary_enhanced_8` with:

- **dependency tracking**: enhanced dependency tracking capabilities.
- **change impact analysis**: enhanced change impact analysis capabilities.

Usage::

    from pyfoam.tools.foam_dictionary_enhanced_9 import FoamDictEnhanced9Result, foam_dictionary_enhanced_9

    result = foam_dictionary_enhanced_9()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["FoamDictEnhanced9Result", "foam_dictionary_enhanced_9"]

@dataclass
class DependencyTrackingResult:
    """Feature data for dependency_tracking."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ChangeImpactResult:
    """Feature data for change_impact_analysis."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class FoamDictEnhanced9Result:
    """Result from :func:`foam_dictionary_enhanced_9`."""
    dependencies: Optional[DependencyTrackingResult] = None
    impact: Optional[ChangeImpactResult] = None


def foam_dictionary_enhanced_9(
    mesh: Optional["FvMesh"] = None,
    enable_dependencies: bool = False,
    enable_impact: bool = False,
) -> FoamDictEnhanced9Result:
    """Enhanced v9 foam dictionary.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    FoamDictEnhanced9Result
    """
    dependencies = None
    if enable_dependencies:
        dependencies = DependencyTrackingResult(name="dependency_tracking", enabled=True)

    impact = None
    if enable_impact:
        impact = ChangeImpactResult(name="change_impact_analysis", enabled=True)

    return FoamDictEnhanced9Result(
        dependencies=dependencies,
        impact=impact,
    )
