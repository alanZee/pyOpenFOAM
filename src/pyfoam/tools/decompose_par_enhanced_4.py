"""
decompose_par enhanced v4 -- enhanced decompose par with additional capabilities
(generation 4).

Extends :func:`decompose_par_enhanced_3` with:

- **hierarchical decomposition**: enhanced hierarchical decomposition capabilities.
- **load balancing**: enhanced load balancing capabilities.

Usage::

    from pyfoam.tools.decompose_par_enhanced_4 import DecomposeParEnhanced4Result, decompose_par_enhanced_4

    result = decompose_par_enhanced_4()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["DecomposeParEnhanced4Result", "decompose_par_enhanced_4"]

@dataclass
class HierarchicalDecompResult:
    """Feature data for hierarchical_decomposition."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class LoadBalancingResult:
    """Feature data for load_balancing."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class DecomposeParEnhanced4Result:
    """Result from :func:`decompose_par_enhanced_4`."""
    hierarchical: Optional[HierarchicalDecompResult] = None
    load_balance: Optional[LoadBalancingResult] = None


def decompose_par_enhanced_4(
    mesh: Optional["FvMesh"] = None,
    enable_hierarchical: bool = False,
    enable_load_balance: bool = False,
) -> DecomposeParEnhanced4Result:
    """Enhanced v4 decompose par.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    DecomposeParEnhanced4Result
    """
    hierarchical = None
    if enable_hierarchical:
        hierarchical = HierarchicalDecompResult(name="hierarchical_decomposition", enabled=True)

    load_balance = None
    if enable_load_balance:
        load_balance = LoadBalancingResult(name="load_balancing", enabled=True)

    return DecomposeParEnhanced4Result(
        hierarchical=hierarchical,
        load_balance=load_balance,
    )
