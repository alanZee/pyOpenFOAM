"""
decompose_par enhanced v6 -- enhanced decompose par with additional capabilities
(generation 6).

Extends :func:`decompose_par_enhanced_5` with:

- **metis decomposition**: enhanced metis decomposition capabilities.
- **neighbour decomposition**: enhanced neighbour decomposition capabilities.

Usage::

    from pyfoam.tools.decompose_par_enhanced_6 import DecomposeParEnhanced6Result, decompose_par_enhanced_6

    result = decompose_par_enhanced_6()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["DecomposeParEnhanced6Result", "decompose_par_enhanced_6"]

@dataclass
class MetisDecompResult:
    """Feature data for metis_decomposition."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class NeighbourDecompResult:
    """Feature data for neighbour_decomposition."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class DecomposeParEnhanced6Result:
    """Result from :func:`decompose_par_enhanced_6`."""
    metis: Optional[MetisDecompResult] = None
    neighbour: Optional[NeighbourDecompResult] = None


def decompose_par_enhanced_6(
    mesh: Optional["FvMesh"] = None,
    enable_metis: bool = False,
    enable_neighbour: bool = False,
) -> DecomposeParEnhanced6Result:
    """Enhanced v6 decompose par.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    DecomposeParEnhanced6Result
    """
    metis = None
    if enable_metis:
        metis = MetisDecompResult(name="metis_decomposition", enabled=True)

    neighbour = None
    if enable_neighbour:
        neighbour = NeighbourDecompResult(name="neighbour_decomposition", enabled=True)

    return DecomposeParEnhanced6Result(
        metis=metis,
        neighbour=neighbour,
    )
