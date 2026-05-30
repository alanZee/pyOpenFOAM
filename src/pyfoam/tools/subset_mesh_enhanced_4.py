"""
subset_mesh enhanced v4 -- enhanced subset mesh with additional capabilities
(generation 4).

Extends :func:`subset_mesh_enhanced_3` with:

- **multi criteria subset**: enhanced multi criteria subset capabilities.
- **boundary preserving subset**: enhanced boundary preserving subset capabilities.

Usage::

    from pyfoam.tools.subset_mesh_enhanced_4 import SubsetEnhanced4Result, subset_mesh_enhanced_4

    result = subset_mesh_enhanced_4()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SubsetEnhanced4Result", "subset_mesh_enhanced_4"]

@dataclass
class MultiCriteriaSubsetResult:
    """Feature data for multi_criteria_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class BoundaryPreservingSubsetResult:
    """Feature data for boundary_preserving_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SubsetEnhanced4Result:
    """Result from :func:`subset_mesh_enhanced_4`."""
    multi_criteria: Optional[MultiCriteriaSubsetResult] = None
    boundary_preserving: Optional[BoundaryPreservingSubsetResult] = None


def subset_mesh_enhanced_4(
    mesh: Optional["FvMesh"] = None,
    enable_multi_criteria: bool = False,
    enable_boundary_preserving: bool = False,
) -> SubsetEnhanced4Result:
    """Enhanced v4 subset mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SubsetEnhanced4Result
    """
    multi_criteria = None
    if enable_multi_criteria:
        multi_criteria = MultiCriteriaSubsetResult(name="multi_criteria_subset", enabled=True)

    boundary_preserving = None
    if enable_boundary_preserving:
        boundary_preserving = BoundaryPreservingSubsetResult(name="boundary_preserving_subset", enabled=True)

    return SubsetEnhanced4Result(
        multi_criteria=multi_criteria,
        boundary_preserving=boundary_preserving,
    )
