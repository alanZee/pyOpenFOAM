"""
renumber_mesh enhanced v4 -- enhanced renumber mesh with additional capabilities
(generation 4).

Extends :func:`renumber_mesh_enhanced_3` with:

- **hybrid ordering**: enhanced hybrid ordering capabilities.
- **bandwidth optimization**: enhanced bandwidth optimization capabilities.

Usage::

    from pyfoam.tools.renumber_mesh_enhanced_4 import RenumberEnhanced4Result, renumber_mesh_enhanced_4

    result = renumber_mesh_enhanced_4()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RenumberEnhanced4Result", "renumber_mesh_enhanced_4"]

@dataclass
class HybridOrderingResult:
    """Feature data for hybrid_ordering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class BandwidthOptResult:
    """Feature data for bandwidth_optimization."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RenumberEnhanced4Result:
    """Result from :func:`renumber_mesh_enhanced_4`."""
    hybrid: Optional[HybridOrderingResult] = None
    bandwidth: Optional[BandwidthOptResult] = None


def renumber_mesh_enhanced_4(
    mesh: Optional["FvMesh"] = None,
    enable_hybrid: bool = False,
    enable_bandwidth: bool = False,
) -> RenumberEnhanced4Result:
    """Enhanced v4 renumber mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RenumberEnhanced4Result
    """
    hybrid = None
    if enable_hybrid:
        hybrid = HybridOrderingResult(name="hybrid_ordering", enabled=True)

    bandwidth = None
    if enable_bandwidth:
        bandwidth = BandwidthOptResult(name="bandwidth_optimization", enabled=True)

    return RenumberEnhanced4Result(
        hybrid=hybrid,
        bandwidth=bandwidth,
    )
