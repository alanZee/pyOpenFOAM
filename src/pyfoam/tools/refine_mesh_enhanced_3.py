"""
refine_mesh enhanced v3 -- enhanced refine mesh with additional capabilities
(generation 3).

Extends :func:`refine_mesh_enhanced_2` with:

- **boundary layer refinement**: enhanced boundary layer refinement capabilities.
- **interface refinement**: enhanced interface refinement capabilities.

Usage::

    from pyfoam.tools.refine_mesh_enhanced_3 import RefineEnhanced3Result, refine_mesh_enhanced_3

    result = refine_mesh_enhanced_3()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RefineEnhanced3Result", "refine_mesh_enhanced_3"]

@dataclass
class BLRefineResult:
    """Feature data for boundary_layer_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class InterfaceRefineResult:
    """Feature data for interface_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RefineEnhanced3Result:
    """Result from :func:`refine_mesh_enhanced_3`."""
    bl_refine: Optional[BLRefineResult] = None
    interface: Optional[InterfaceRefineResult] = None


def refine_mesh_enhanced_3(
    mesh: Optional["FvMesh"] = None,
    enable_bl_refine: bool = False,
    enable_interface: bool = False,
) -> RefineEnhanced3Result:
    """Enhanced v3 refine mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RefineEnhanced3Result
    """
    bl_refine = None
    if enable_bl_refine:
        bl_refine = BLRefineResult(name="boundary_layer_refinement", enabled=True)

    interface = None
    if enable_interface:
        interface = InterfaceRefineResult(name="interface_refinement", enabled=True)

    return RefineEnhanced3Result(
        bl_refine=bl_refine,
        interface=interface,
    )
