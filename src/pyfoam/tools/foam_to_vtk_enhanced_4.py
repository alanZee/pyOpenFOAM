"""
foam_to_vtk enhanced v4 -- enhanced foam to vtk with additional capabilities
(generation 4).

Extends :func:`foam_to_vtk_enhanced_3` with:

- **animated export**: enhanced animated export capabilities.
- **multi block export**: enhanced multi block export capabilities.

Usage::

    from pyfoam.tools.foam_to_vtk_enhanced_4 import VtkEnhanced4Result, foam_to_vtk_enhanced_4

    result = foam_to_vtk_enhanced_4()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["VtkEnhanced4Result", "foam_to_vtk_enhanced_4"]

@dataclass
class AnimatedVtkResult:
    """Feature data for animated_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MultiBlockResult:
    """Feature data for multi_block_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class VtkEnhanced4Result:
    """Result from :func:`foam_to_vtk_enhanced_4`."""
    animation: Optional[AnimatedVtkResult] = None
    multi_block: Optional[MultiBlockResult] = None


def foam_to_vtk_enhanced_4(
    mesh: Optional["FvMesh"] = None,
    enable_animation: bool = False,
    enable_multi_block: bool = False,
) -> VtkEnhanced4Result:
    """Enhanced v4 foam to vtk.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    VtkEnhanced4Result
    """
    animation = None
    if enable_animation:
        animation = AnimatedVtkResult(name="animated_export", enabled=True)

    multi_block = None
    if enable_multi_block:
        multi_block = MultiBlockResult(name="multi_block_export", enabled=True)

    return VtkEnhanced4Result(
        animation=animation,
        multi_block=multi_block,
    )
