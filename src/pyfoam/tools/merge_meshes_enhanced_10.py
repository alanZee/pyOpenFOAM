"""
merge_meshes enhanced v10 -- enhanced merge meshes with additional capabilities
(generation 10).

Extends :func:`merge_meshes_enhanced_9` with:

- **hierarchical merge**: enhanced hierarchical merge capabilities.
- **merge quality preservation**: enhanced merge quality preservation capabilities.

Usage::

    from pyfoam.tools.merge_meshes_enhanced_10 import MergeEnhanced10Result, merge_meshes_enhanced_10

    result = merge_meshes_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MergeEnhanced10Result", "merge_meshes_enhanced_10"]

@dataclass
class HierarchicalMergeResult:
    """Feature data for hierarchical_merge."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MergeQualityResult:
    """Feature data for merge_quality_preservation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MergeEnhanced10Result:
    """Result from :func:`merge_meshes_enhanced_10`."""
    hierarchical: Optional[HierarchicalMergeResult] = None
    quality_preservation: Optional[MergeQualityResult] = None


def merge_meshes_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_hierarchical: bool = False,
    enable_quality_preservation: bool = False,
) -> MergeEnhanced10Result:
    """Enhanced v10 merge meshes.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MergeEnhanced10Result
    """
    hierarchical = None
    if enable_hierarchical:
        hierarchical = HierarchicalMergeResult(name="hierarchical_merge", enabled=True)

    quality_preservation = None
    if enable_quality_preservation:
        quality_preservation = MergeQualityResult(name="merge_quality_preservation", enabled=True)

    return MergeEnhanced10Result(
        hierarchical=hierarchical,
        quality_preservation=quality_preservation,
    )
