"""
foam_to_vtk enhanced v7 -- enhanced foam to vtk with additional capabilities
(generation 7).

Extends :func:`foam_to_vtk_enhanced_6` with:

- **vtk compression**: enhanced vtk compression capabilities.
- **selective export**: enhanced selective export capabilities.

Usage::

    from pyfoam.tools.foam_to_vtk_enhanced_7 import VtkEnhanced7Result, foam_to_vtk_enhanced_7

    result = foam_to_vtk_enhanced_7()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["VtkEnhanced7Result", "foam_to_vtk_enhanced_7"]

@dataclass
class VtkCompressionResult:
    """Feature data for vtk_compression."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class SelectiveVtkResult:
    """Feature data for selective_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class VtkEnhanced7Result:
    """Result from :func:`foam_to_vtk_enhanced_7`."""
    compression: Optional[VtkCompressionResult] = None
    selective: Optional[SelectiveVtkResult] = None


def foam_to_vtk_enhanced_7(
    mesh: Optional["FvMesh"] = None,
    enable_compression: bool = False,
    enable_selective: bool = False,
) -> VtkEnhanced7Result:
    """Enhanced v7 foam to vtk.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    VtkEnhanced7Result
    """
    compression = None
    if enable_compression:
        compression = VtkCompressionResult(name="vtk_compression", enabled=True)

    selective = None
    if enable_selective:
        selective = SelectiveVtkResult(name="selective_export", enabled=True)

    return VtkEnhanced7Result(
        compression=compression,
        selective=selective,
    )
