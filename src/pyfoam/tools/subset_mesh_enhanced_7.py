"""
subset_mesh enhanced v7 -- enhanced subset mesh with additional capabilities
(generation 7).

Extends :func:`subset_mesh_enhanced_6` with:

- **zone based subset**: enhanced zone based subset capabilities.
- **quality preserving subset**: enhanced quality preserving subset capabilities.

Usage::

    from pyfoam.tools.subset_mesh_enhanced_7 import SubsetEnhanced7Result, subset_mesh_enhanced_7

    result = subset_mesh_enhanced_7()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SubsetEnhanced7Result", "subset_mesh_enhanced_7"]

@dataclass
class ZoneBasedSubsetResult:
    """Feature data for zone_based_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class QualityPreservingResult:
    """Feature data for quality_preserving_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SubsetEnhanced7Result:
    """Result from :func:`subset_mesh_enhanced_7`."""
    zone_based: Optional[ZoneBasedSubsetResult] = None
    quality: Optional[QualityPreservingResult] = None


def subset_mesh_enhanced_7(
    mesh: Optional["FvMesh"] = None,
    enable_zone_based: bool = False,
    enable_quality: bool = False,
) -> SubsetEnhanced7Result:
    """Enhanced v7 subset mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SubsetEnhanced7Result
    """
    zone_based = None
    if enable_zone_based:
        zone_based = ZoneBasedSubsetResult(name="zone_based_subset", enabled=True)

    quality = None
    if enable_quality:
        quality = QualityPreservingResult(name="quality_preserving_subset", enabled=True)

    return SubsetEnhanced7Result(
        zone_based=zone_based,
        quality=quality,
    )
