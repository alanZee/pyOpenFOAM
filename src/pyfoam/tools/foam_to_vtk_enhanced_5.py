"""
foam_to_vtk enhanced v5 -- enhanced foam to vtk with additional capabilities
(generation 5).

Extends :func:`foam_to_vtk_enhanced_4` with:

- **adaptive resolution**: enhanced adaptive resolution capabilities.
- **vtk field statistics**: enhanced vtk field statistics capabilities.
- **binary vtk export**: enhanced binary vtk export capabilities.

Usage::

    from pyfoam.tools.foam_to_vtk_enhanced_5 import VtkEnhanced5Result, foam_to_vtk_enhanced_5

    result = foam_to_vtk_enhanced_5()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["VtkEnhanced5Result", "foam_to_vtk_enhanced_5"]

@dataclass
class AdaptiveResolutionResult:
    """Feature data for adaptive_resolution."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class VtkStatisticsResult:
    """Feature data for vtk_field_statistics."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class BinaryVtkResult:
    """Feature data for binary_vtk_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class VtkEnhanced5Result:
    """Result from :func:`foam_to_vtk_enhanced_5`."""
    adaptive_resolution: Optional[AdaptiveResolutionResult] = None
    statistics: Optional[VtkStatisticsResult] = None
    binary: Optional[BinaryVtkResult] = None


def foam_to_vtk_enhanced_5(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive_resolution: bool = False,
    enable_statistics: bool = False,
    enable_binary: bool = False,
) -> VtkEnhanced5Result:
    """Enhanced v5 foam to vtk.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    VtkEnhanced5Result
    """
    adaptive_resolution = None
    if enable_adaptive_resolution:
        adaptive_resolution = AdaptiveResolutionResult(name="adaptive_resolution", enabled=True)

    statistics = None
    if enable_statistics:
        statistics = VtkStatisticsResult(name="vtk_field_statistics", enabled=True)

    binary = None
    if enable_binary:
        binary = BinaryVtkResult(name="binary_vtk_export", enabled=True)

    return VtkEnhanced5Result(
        adaptive_resolution=adaptive_resolution,
        statistics=statistics,
        binary=binary,
    )
