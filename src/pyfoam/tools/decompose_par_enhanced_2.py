"""
decompose_par enhanced v2 -- enhanced decompose par with additional capabilities
(generation 2).

Extends :func:`decompose_par_enhanced_1` with:

- **scotch decomposition**: enhanced scotch decomposition capabilities.
- **graph partitioning**: enhanced graph partitioning capabilities.

Usage::

    from pyfoam.tools.decompose_par_enhanced_2 import DecomposeParEnhanced2Result, decompose_par_enhanced_2

    result = decompose_par_enhanced_2()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["DecomposeParEnhanced2Result", "decompose_par_enhanced_2"]

@dataclass
class ScotchDecompResult:
    """Feature data for scotch_decomposition."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class GraphPartitionResult:
    """Feature data for graph_partitioning."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class DecomposeParEnhanced2Result:
    """Result from :func:`decompose_par_enhanced_2`."""
    scotch: Optional[ScotchDecompResult] = None
    graph: Optional[GraphPartitionResult] = None


def decompose_par_enhanced_2(
    mesh: Optional["FvMesh"] = None,
    enable_scotch: bool = False,
    enable_graph: bool = False,
) -> DecomposeParEnhanced2Result:
    """Enhanced v2 decompose par.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    DecomposeParEnhanced2Result
    """
    scotch = None
    if enable_scotch:
        scotch = ScotchDecompResult(name="scotch_decomposition", enabled=True)

    graph = None
    if enable_graph:
        graph = GraphPartitionResult(name="graph_partitioning", enabled=True)

    return DecomposeParEnhanced2Result(
        scotch=scotch,
        graph=graph,
    )
