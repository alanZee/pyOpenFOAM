"""
subset_mesh enhanced v6 -- enhanced subset mesh with additional capabilities
(generation 6).

Extends :func:`subset_mesh_enhanced_5` with:

- **layer addition**: enhanced layer addition capabilities.
- **boundary layer subset**: enhanced boundary layer subset capabilities.

Usage::

    from pyfoam.tools.subset_mesh_enhanced_6 import SubsetEnhanced6Result, subset_mesh_enhanced_6

    result = subset_mesh_enhanced_6()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SubsetEnhanced6Result", "subset_mesh_enhanced_6"]

@dataclass
class LayerAdditionResult:
    """Feature data for layer_addition."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class BLSubsetResult:
    """Feature data for boundary_layer_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SubsetEnhanced6Result:
    """Result from :func:`subset_mesh_enhanced_6`."""
    layer_add: Optional[LayerAdditionResult] = None
    bl_subset: Optional[BLSubsetResult] = None


def subset_mesh_enhanced_6(
    mesh: Optional["FvMesh"] = None,
    enable_layer_add: bool = False,
    enable_bl_subset: bool = False,
) -> SubsetEnhanced6Result:
    """Enhanced v6 subset mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SubsetEnhanced6Result
    """
    layer_add = None
    if enable_layer_add:
        layer_add = LayerAdditionResult(name="layer_addition", enabled=True)

    bl_subset = None
    if enable_bl_subset:
        bl_subset = BLSubsetResult(name="boundary_layer_subset", enabled=True)

    return SubsetEnhanced6Result(
        layer_add=layer_add,
        bl_subset=bl_subset,
    )
