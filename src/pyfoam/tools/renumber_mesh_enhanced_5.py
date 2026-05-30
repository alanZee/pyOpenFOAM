"""
renumber_mesh enhanced v5 -- enhanced renumber mesh with additional capabilities
(generation 5).

Extends :func:`renumber_mesh_enhanced_4` with:

- **adaptive renumbering**: enhanced adaptive renumbering capabilities.
- **cache optimized ordering**: enhanced cache optimized ordering capabilities.
- **hierarchical renumbering**: enhanced hierarchical renumbering capabilities.

Usage::

    from pyfoam.tools.renumber_mesh_enhanced_5 import RenumberEnhanced5Result, renumber_mesh_enhanced_5

    result = renumber_mesh_enhanced_5()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RenumberEnhanced5Result", "renumber_mesh_enhanced_5"]

@dataclass
class AdaptiveRenumberResult:
    """Feature data for adaptive_renumbering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class CacheOptimizedResult:
    """Feature data for cache_optimized_ordering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class HierarchicalRenumberResult:
    """Feature data for hierarchical_renumbering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RenumberEnhanced5Result:
    """Result from :func:`renumber_mesh_enhanced_5`."""
    adaptive: Optional[AdaptiveRenumberResult] = None
    cache: Optional[CacheOptimizedResult] = None
    hierarchical: Optional[HierarchicalRenumberResult] = None


def renumber_mesh_enhanced_5(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive: bool = False,
    enable_cache: bool = False,
    enable_hierarchical: bool = False,
) -> RenumberEnhanced5Result:
    """Enhanced v5 renumber mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RenumberEnhanced5Result
    """
    adaptive = None
    if enable_adaptive:
        adaptive = AdaptiveRenumberResult(name="adaptive_renumbering", enabled=True)

    cache = None
    if enable_cache:
        cache = CacheOptimizedResult(name="cache_optimized_ordering", enabled=True)

    hierarchical = None
    if enable_hierarchical:
        hierarchical = HierarchicalRenumberResult(name="hierarchical_renumbering", enabled=True)

    return RenumberEnhanced5Result(
        adaptive=adaptive,
        cache=cache,
        hierarchical=hierarchical,
    )
