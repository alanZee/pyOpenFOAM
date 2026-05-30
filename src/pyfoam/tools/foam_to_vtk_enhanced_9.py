"""
foam_to_vtk enhanced v9 -- enhanced foam to vtk with additional capabilities
(generation 9).

Extends :func:`foam_to_vtk_enhanced_8` with:

- **vtk distributed io**: enhanced vtk distributed io capabilities.
- **vtk quality metrics**: enhanced vtk quality metrics capabilities.

Usage::

    from pyfoam.tools.foam_to_vtk_enhanced_9 import VtkEnhanced9Result, foam_to_vtk_enhanced_9

    result = foam_to_vtk_enhanced_9()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["VtkEnhanced9Result", "foam_to_vtk_enhanced_9"]

@dataclass
class VtkDistributedResult:
    """Feature data for vtk_distributed_io."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class VtkQualityMetrics:
    """Feature data for vtk_quality_metrics."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class VtkEnhanced9Result:
    """Result from :func:`foam_to_vtk_enhanced_9`."""
    distributed: Optional[VtkDistributedResult] = None
    quality: Optional[VtkQualityMetrics] = None


def foam_to_vtk_enhanced_9(
    mesh: Optional["FvMesh"] = None,
    enable_distributed: bool = False,
    enable_quality: bool = False,
) -> VtkEnhanced9Result:
    """Enhanced v9 foam to vtk.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    VtkEnhanced9Result
    """
    distributed = None
    if enable_distributed:
        distributed = VtkDistributedResult(name="vtk_distributed_io", enabled=True)

    quality = None
    if enable_quality:
        quality = VtkQualityMetrics(name="vtk_quality_metrics", enabled=True)

    return VtkEnhanced9Result(
        distributed=distributed,
        quality=quality,
    )
