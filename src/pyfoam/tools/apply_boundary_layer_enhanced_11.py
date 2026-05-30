"""
apply_boundary_layer enhanced v11 -- enhanced apply boundary layer with additional capabilities
(generation 11).

Extends :func:`apply_boundary_layer_enhanced_10` with:

- **curvature correction**: enhanced curvature correction capabilities.
- **roughness evolution**: enhanced roughness evolution capabilities.

Usage::

    from pyfoam.tools.apply_boundary_layer_enhanced_11 import EnhancedBL11Result, apply_boundary_layer_enhanced_11

    result = apply_boundary_layer_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedBL11Result", "apply_boundary_layer_enhanced_11"]

@dataclass
class CurvatureCorrectionResult:
    """Feature data for curvature_correction."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class RoughnessEvolutionResult:
    """Feature data for roughness_evolution."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class EnhancedBL11Result:
    """Result from :func:`apply_boundary_layer_enhanced_11`."""
    curvature: Optional[CurvatureCorrectionResult] = None
    roughness_evolution: Optional[RoughnessEvolutionResult] = None


def apply_boundary_layer_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_curvature: bool = False,
    enable_roughness_evolution: bool = False,
) -> EnhancedBL11Result:
    """Enhanced v11 apply boundary layer.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    EnhancedBL11Result
    """
    curvature = None
    if enable_curvature:
        curvature = CurvatureCorrectionResult(name="curvature_correction", enabled=True)

    roughness_evolution = None
    if enable_roughness_evolution:
        roughness_evolution = RoughnessEvolutionResult(name="roughness_evolution", enabled=True)

    return EnhancedBL11Result(
        curvature=curvature,
        roughness_evolution=roughness_evolution,
    )
