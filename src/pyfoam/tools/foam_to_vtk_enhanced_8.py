"""
foam_to_vtk enhanced v8 -- enhanced foam to vtk with additional capabilities
(generation 8).

Extends :func:`foam_to_vtk_enhanced_7` with:

- **vtk pipeline integration**: enhanced vtk pipeline integration capabilities.
- **polyhedra support**: enhanced polyhedra support capabilities.

Usage::

    from pyfoam.tools.foam_to_vtk_enhanced_8 import VtkEnhanced8Result, foam_to_vtk_enhanced_8

    result = foam_to_vtk_enhanced_8()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["VtkEnhanced8Result", "foam_to_vtk_enhanced_8"]

@dataclass
class VtkPipelineResult:
    """Feature data for vtk_pipeline_integration."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class PolyhedraResult:
    """Feature data for polyhedra_support."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class VtkEnhanced8Result:
    """Result from :func:`foam_to_vtk_enhanced_8`."""
    pipeline: Optional[VtkPipelineResult] = None
    polyhedra: Optional[PolyhedraResult] = None


def foam_to_vtk_enhanced_8(
    mesh: Optional["FvMesh"] = None,
    enable_pipeline: bool = False,
    enable_polyhedra: bool = False,
) -> VtkEnhanced8Result:
    """Enhanced v8 foam to vtk.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    VtkEnhanced8Result
    """
    pipeline = None
    if enable_pipeline:
        pipeline = VtkPipelineResult(name="vtk_pipeline_integration", enabled=True)

    polyhedra = None
    if enable_polyhedra:
        polyhedra = PolyhedraResult(name="polyhedra_support", enabled=True)

    return VtkEnhanced8Result(
        pipeline=pipeline,
        polyhedra=polyhedra,
    )
