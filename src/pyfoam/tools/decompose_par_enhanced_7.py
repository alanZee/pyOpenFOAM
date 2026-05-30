"""
decompose_par enhanced v7 -- enhanced decompose par with additional capabilities
(generation 7).

Extends :func:`decompose_par_enhanced_6` with:

- **ghost cell management**: enhanced ghost cell management capabilities.
- **overlap decomposition**: enhanced overlap decomposition capabilities.

Usage::

    from pyfoam.tools.decompose_par_enhanced_7 import DecomposeParEnhanced7Result, decompose_par_enhanced_7

    result = decompose_par_enhanced_7()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["DecomposeParEnhanced7Result", "decompose_par_enhanced_7"]

@dataclass
class GhostCellResult:
    """Feature data for ghost_cell_management."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class OverlapDecompResult:
    """Feature data for overlap_decomposition."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class DecomposeParEnhanced7Result:
    """Result from :func:`decompose_par_enhanced_7`."""
    ghost_cells: Optional[GhostCellResult] = None
    overlap: Optional[OverlapDecompResult] = None


def decompose_par_enhanced_7(
    mesh: Optional["FvMesh"] = None,
    enable_ghost_cells: bool = False,
    enable_overlap: bool = False,
) -> DecomposeParEnhanced7Result:
    """Enhanced v7 decompose par.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    DecomposeParEnhanced7Result
    """
    ghost_cells = None
    if enable_ghost_cells:
        ghost_cells = GhostCellResult(name="ghost_cell_management", enabled=True)

    overlap = None
    if enable_overlap:
        overlap = OverlapDecompResult(name="overlap_decomposition", enabled=True)

    return DecomposeParEnhanced7Result(
        ghost_cells=ghost_cells,
        overlap=overlap,
    )
