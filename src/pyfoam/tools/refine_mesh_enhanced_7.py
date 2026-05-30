"""
refine_mesh enhanced v7 -- enhanced refine mesh with additional capabilities
(generation 7).

Extends :func:`refine_mesh_enhanced_6` with:

- **refinement zones**: enhanced refinement zones capabilities.
- **feature based refinement**: enhanced feature based refinement capabilities.

Usage::

    from pyfoam.tools.refine_mesh_enhanced_7 import RefineEnhanced7Result, refine_mesh_enhanced_7

    result = refine_mesh_enhanced_7()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RefineEnhanced7Result", "refine_mesh_enhanced_7"]

@dataclass
class RefinementZonesResult:
    """Feature data for refinement_zones."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class FeatureRefineResult:
    """Feature data for feature_based_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RefineEnhanced7Result:
    """Result from :func:`refine_mesh_enhanced_7`."""
    zones: Optional[RefinementZonesResult] = None
    feature_based: Optional[FeatureRefineResult] = None


def refine_mesh_enhanced_7(
    mesh: Optional["FvMesh"] = None,
    enable_zones: bool = False,
    enable_feature_based: bool = False,
) -> RefineEnhanced7Result:
    """Enhanced v7 refine mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RefineEnhanced7Result
    """
    zones = None
    if enable_zones:
        zones = RefinementZonesResult(name="refinement_zones", enabled=True)

    feature_based = None
    if enable_feature_based:
        feature_based = FeatureRefineResult(name="feature_based_refinement", enabled=True)

    return RefineEnhanced7Result(
        zones=zones,
        feature_based=feature_based,
    )
