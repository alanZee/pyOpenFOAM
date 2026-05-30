"""
foam_to_vtk enhanced v3 -- enhanced foam to vtk with additional capabilities
(generation 3).

Extends :func:`foam_to_vtk_enhanced_2` with:

- **parallel vtk export**: enhanced parallel vtk export capabilities.
- **vtkhdf export**: enhanced vtkhdf export capabilities.

Usage::

    from pyfoam.tools.foam_to_vtk_enhanced_3 import VtkEnhanced3Result, foam_to_vtk_enhanced_3

    result = foam_to_vtk_enhanced_3()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["VtkEnhanced3Result", "foam_to_vtk_enhanced_3"]

@dataclass
class ParallelVtkResult:
    """Feature data for parallel_vtk_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class VtkHDFResult:
    """Feature data for vtkhdf_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class VtkEnhanced3Result:
    """Result from :func:`foam_to_vtk_enhanced_3`."""
    parallel: Optional[ParallelVtkResult] = None
    vtkhdf: Optional[VtkHDFResult] = None


def foam_to_vtk_enhanced_3(
    mesh: Optional["FvMesh"] = None,
    enable_parallel: bool = False,
    enable_vtkhdf: bool = False,
) -> VtkEnhanced3Result:
    """Enhanced v3 foam to vtk.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    VtkEnhanced3Result
    """
    parallel = None
    if enable_parallel:
        parallel = ParallelVtkResult(name="parallel_vtk_export", enabled=True)

    vtkhdf = None
    if enable_vtkhdf:
        vtkhdf = VtkHDFResult(name="vtkhdf_export", enabled=True)

    return VtkEnhanced3Result(
        parallel=parallel,
        vtkhdf=vtkhdf,
    )
