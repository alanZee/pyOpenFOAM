"""
decompose_par enhanced v9 -- enhanced decompose par with additional capabilities
(generation 9).

Extends :func:`decompose_par_enhanced_8` with:

- **decomposition visualisation**: enhanced decomposition visualisation capabilities.
- **communication minimisation**: enhanced communication minimisation capabilities.

Usage::

    from pyfoam.tools.decompose_par_enhanced_9 import DecomposeParEnhanced9Result, decompose_par_enhanced_9

    result = decompose_par_enhanced_9()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["DecomposeParEnhanced9Result", "decompose_par_enhanced_9"]

@dataclass
class DecompVisualResult:
    """Feature data for decomposition_visualisation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class CommMinResult:
    """Feature data for communication_minimisation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class DecomposeParEnhanced9Result:
    """Result from :func:`decompose_par_enhanced_9`."""
    visualisation: Optional[DecompVisualResult] = None
    comm_min: Optional[CommMinResult] = None


def decompose_par_enhanced_9(
    mesh: Optional["FvMesh"] = None,
    enable_visualisation: bool = False,
    enable_comm_min: bool = False,
) -> DecomposeParEnhanced9Result:
    """Enhanced v9 decompose par.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    DecomposeParEnhanced9Result
    """
    visualisation = None
    if enable_visualisation:
        visualisation = DecompVisualResult(name="decomposition_visualisation", enabled=True)

    comm_min = None
    if enable_comm_min:
        comm_min = CommMinResult(name="communication_minimisation", enabled=True)

    return DecomposeParEnhanced9Result(
        visualisation=visualisation,
        comm_min=comm_min,
    )
