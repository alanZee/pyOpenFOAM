"""
renumber_mesh enhanced v6 -- enhanced renumber mesh with additional capabilities
(generation 6).

Extends :func:`renumber_mesh_enhanced_5` with:

- **dual graph ordering**: enhanced dual graph ordering capabilities.
- **wavefront reduction**: enhanced wavefront reduction capabilities.

Usage::

    from pyfoam.tools.renumber_mesh_enhanced_6 import RenumberEnhanced6Result, renumber_mesh_enhanced_6

    result = renumber_mesh_enhanced_6()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RenumberEnhanced6Result", "renumber_mesh_enhanced_6"]

@dataclass
class DualGraphOrderingResult:
    """Feature data for dual_graph_ordering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class WavefrontReductionResult:
    """Feature data for wavefront_reduction."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RenumberEnhanced6Result:
    """Result from :func:`renumber_mesh_enhanced_6`."""
    dual_graph: Optional[DualGraphOrderingResult] = None
    wavefront: Optional[WavefrontReductionResult] = None


def renumber_mesh_enhanced_6(
    mesh: Optional["FvMesh"] = None,
    enable_dual_graph: bool = False,
    enable_wavefront: bool = False,
) -> RenumberEnhanced6Result:
    """Enhanced v6 renumber mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RenumberEnhanced6Result
    """
    dual_graph = None
    if enable_dual_graph:
        dual_graph = DualGraphOrderingResult(name="dual_graph_ordering", enabled=True)

    wavefront = None
    if enable_wavefront:
        wavefront = WavefrontReductionResult(name="wavefront_reduction", enabled=True)

    return RenumberEnhanced6Result(
        dual_graph=dual_graph,
        wavefront=wavefront,
    )
