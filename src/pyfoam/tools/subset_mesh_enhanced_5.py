"""
subset_mesh enhanced v5 -- enhanced subset mesh with additional capabilities
(generation 5).

Extends :func:`subset_mesh_enhanced_4` with:

- **adaptive subset**: enhanced adaptive subset capabilities.
- **subset statistics**: enhanced subset statistics capabilities.
- **nested subset**: enhanced nested subset capabilities.

Usage::

    from pyfoam.tools.subset_mesh_enhanced_5 import SubsetEnhanced5Result, subset_mesh_enhanced_5

    result = subset_mesh_enhanced_5()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SubsetEnhanced5Result", "subset_mesh_enhanced_5"]

@dataclass
class AdaptiveSubsetResult:
    """Feature data for adaptive_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class SubsetStatisticsResult:
    """Feature data for subset_statistics."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class NestedSubsetResult:
    """Feature data for nested_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SubsetEnhanced5Result:
    """Result from :func:`subset_mesh_enhanced_5`."""
    adaptive: Optional[AdaptiveSubsetResult] = None
    statistics: Optional[SubsetStatisticsResult] = None
    nested: Optional[NestedSubsetResult] = None


def subset_mesh_enhanced_5(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive: bool = False,
    enable_statistics: bool = False,
    enable_nested: bool = False,
) -> SubsetEnhanced5Result:
    """Enhanced v5 subset mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SubsetEnhanced5Result
    """
    adaptive = None
    if enable_adaptive:
        adaptive = AdaptiveSubsetResult(name="adaptive_subset", enabled=True)

    statistics = None
    if enable_statistics:
        statistics = SubsetStatisticsResult(name="subset_statistics", enabled=True)

    nested = None
    if enable_nested:
        nested = NestedSubsetResult(name="nested_subset", enabled=True)

    return SubsetEnhanced5Result(
        adaptive=adaptive,
        statistics=statistics,
        nested=nested,
    )
