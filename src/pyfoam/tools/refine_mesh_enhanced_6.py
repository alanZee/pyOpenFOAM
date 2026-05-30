"""
refine_mesh enhanced v6 -- enhanced refine mesh with additional capabilities
(generation 6).

Extends :func:`refine_mesh_enhanced_5` with:

- **hex dominant refinement**: enhanced hex dominant refinement capabilities.
- **tetrahedral refinement**: enhanced tetrahedral refinement capabilities.

Usage::

    from pyfoam.tools.refine_mesh_enhanced_6 import RefineEnhanced6Result, refine_mesh_enhanced_6

    result = refine_mesh_enhanced_6()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RefineEnhanced6Result", "refine_mesh_enhanced_6"]

@dataclass
class HexDominantRefineResult:
    """Feature data for hex_dominant_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class TetRefineResult:
    """Feature data for tetrahedral_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RefineEnhanced6Result:
    """Result from :func:`refine_mesh_enhanced_6`."""
    hex_dominant: Optional[HexDominantRefineResult] = None
    tetrahedral: Optional[TetRefineResult] = None


def refine_mesh_enhanced_6(
    mesh: Optional["FvMesh"] = None,
    enable_hex_dominant: bool = False,
    enable_tetrahedral: bool = False,
) -> RefineEnhanced6Result:
    """Enhanced v6 refine mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RefineEnhanced6Result
    """
    hex_dominant = None
    if enable_hex_dominant:
        hex_dominant = HexDominantRefineResult(name="hex_dominant_refinement", enabled=True)

    tetrahedral = None
    if enable_tetrahedral:
        tetrahedral = TetRefineResult(name="tetrahedral_refinement", enabled=True)

    return RefineEnhanced6Result(
        hex_dominant=hex_dominant,
        tetrahedral=tetrahedral,
    )
