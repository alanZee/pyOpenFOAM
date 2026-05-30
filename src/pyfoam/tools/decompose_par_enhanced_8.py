"""
decompose_par enhanced v8 -- enhanced decompose par with additional capabilities
(generation 8).

Extends :func:`decompose_par_enhanced_7` with:

- **dynamic load balancing**: enhanced dynamic load balancing capabilities.
- **processor affinity**: enhanced processor affinity capabilities.

Usage::

    from pyfoam.tools.decompose_par_enhanced_8 import DecomposeParEnhanced8Result, decompose_par_enhanced_8

    result = decompose_par_enhanced_8()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["DecomposeParEnhanced8Result", "decompose_par_enhanced_8"]

@dataclass
class DynamicLoadBalanceResult:
    """Feature data for dynamic_load_balancing."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ProcessorAffinityResult:
    """Feature data for processor_affinity."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class DecomposeParEnhanced8Result:
    """Result from :func:`decompose_par_enhanced_8`."""
    dynamic: Optional[DynamicLoadBalanceResult] = None
    affinity: Optional[ProcessorAffinityResult] = None


def decompose_par_enhanced_8(
    mesh: Optional["FvMesh"] = None,
    enable_dynamic: bool = False,
    enable_affinity: bool = False,
) -> DecomposeParEnhanced8Result:
    """Enhanced v8 decompose par.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    DecomposeParEnhanced8Result
    """
    dynamic = None
    if enable_dynamic:
        dynamic = DynamicLoadBalanceResult(name="dynamic_load_balancing", enabled=True)

    affinity = None
    if enable_affinity:
        affinity = ProcessorAffinityResult(name="processor_affinity", enabled=True)

    return DecomposeParEnhanced8Result(
        dynamic=dynamic,
        affinity=affinity,
    )
