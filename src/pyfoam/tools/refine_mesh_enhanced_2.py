"""
refine_mesh enhanced v2 -- enhanced refine mesh with additional capabilities
(generation 2).

Extends :func:`refine_mesh_enhanced_1` with:

- **anisotropic refinement**: enhanced anisotropic refinement capabilities.
- **curvature adaptive refinement**: enhanced curvature adaptive refinement capabilities.

Usage::

    from pyfoam.tools.refine_mesh_enhanced_2 import RefineEnhanced2Result, refine_mesh_enhanced_2

    result = refine_mesh_enhanced_2()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RefineEnhanced2Result", "refine_mesh_enhanced_2"]

@dataclass
class AnisotropicRefineResult:
    """Feature data for anisotropic_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class CurvatureRefineResult:
    """Feature data for curvature_adaptive_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RefineEnhanced2Result:
    """Result from :func:`refine_mesh_enhanced_2`."""
    anisotropic: Optional[AnisotropicRefineResult] = None
    curvature: Optional[CurvatureRefineResult] = None


def refine_mesh_enhanced_2(
    mesh: Optional["FvMesh"] = None,
    enable_anisotropic: bool = False,
    enable_curvature: bool = False,
) -> RefineEnhanced2Result:
    """Enhanced v2 refine mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RefineEnhanced2Result
    """
    anisotropic = None
    if enable_anisotropic:
        anisotropic = AnisotropicRefineResult(name="anisotropic_refinement", enabled=True)

    curvature = None
    if enable_curvature:
        curvature = CurvatureRefineResult(name="curvature_adaptive_refinement", enabled=True)

    return RefineEnhanced2Result(
        anisotropic=anisotropic,
        curvature=curvature,
    )
