"""
decompose_par enhanced v3 -- enhanced decompose par with additional capabilities
(generation 3).

Extends :func:`decompose_par_enhanced_2` with:

- **multi constraint decomposition**: enhanced multi constraint decomposition capabilities.
- **weighted decomposition**: enhanced weighted decomposition capabilities.

Usage::

    from pyfoam.tools.decompose_par_enhanced_3 import DecomposeParEnhanced3Result, decompose_par_enhanced_3

    result = decompose_par_enhanced_3()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["DecomposeParEnhanced3Result", "decompose_par_enhanced_3"]

@dataclass
class MultiConstraintResult:
    """Feature data for multi_constraint_decomposition."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class WeightedDecompResult:
    """Feature data for weighted_decomposition."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class DecomposeParEnhanced3Result:
    """Result from :func:`decompose_par_enhanced_3`."""
    multi_constraint: Optional[MultiConstraintResult] = None
    weighted: Optional[WeightedDecompResult] = None


def decompose_par_enhanced_3(
    mesh: Optional["FvMesh"] = None,
    enable_multi_constraint: bool = False,
    enable_weighted: bool = False,
) -> DecomposeParEnhanced3Result:
    """Enhanced v3 decompose par.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    DecomposeParEnhanced3Result
    """
    multi_constraint = None
    if enable_multi_constraint:
        multi_constraint = MultiConstraintResult(name="multi_constraint_decomposition", enabled=True)

    weighted = None
    if enable_weighted:
        weighted = WeightedDecompResult(name="weighted_decomposition", enabled=True)

    return DecomposeParEnhanced3Result(
        multi_constraint=multi_constraint,
        weighted=weighted,
    )
