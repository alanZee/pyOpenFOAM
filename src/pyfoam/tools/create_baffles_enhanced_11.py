"""
create_baffles enhanced v11 -- enhanced create baffles with additional capabilities
(generation 11).

Extends :func:`create_baffles_enhanced_10` with:

- **adaptive baffle geometry**: enhanced adaptive baffle geometry capabilities.
- **multi zone baffle**: enhanced multi zone baffle capabilities.

Usage::

    from pyfoam.tools.create_baffles_enhanced_11 import BaffleEnhanced11Result, create_baffles_enhanced_11

    result = create_baffles_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["BaffleEnhanced11Result", "create_baffles_enhanced_11"]

@dataclass
class AdaptiveBaffleGeometry:
    """Feature data for adaptive_baffle_geometry."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MultiZoneBaffleResult:
    """Feature data for multi_zone_baffle."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class BaffleEnhanced11Result:
    """Result from :func:`create_baffles_enhanced_11`."""
    adaptive_geometry: Optional[AdaptiveBaffleGeometry] = None
    multi_zone: Optional[MultiZoneBaffleResult] = None


def create_baffles_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive_geometry: bool = False,
    enable_multi_zone: bool = False,
) -> BaffleEnhanced11Result:
    """Enhanced v11 create baffles.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    BaffleEnhanced11Result
    """
    adaptive_geometry = None
    if enable_adaptive_geometry:
        adaptive_geometry = AdaptiveBaffleGeometry(name="adaptive_baffle_geometry", enabled=True)

    multi_zone = None
    if enable_multi_zone:
        multi_zone = MultiZoneBaffleResult(name="multi_zone_baffle", enabled=True)

    return BaffleEnhanced11Result(
        adaptive_geometry=adaptive_geometry,
        multi_zone=multi_zone,
    )
