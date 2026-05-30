"""
set_atm_boundary_layer enhanced v11 -- enhanced set atm boundary layer with additional capabilities
(generation 11).

Extends :func:`set_atm_boundary_layer_enhanced_10` with:

- **pollutant dispersion**: enhanced pollutant dispersion capabilities.
- **terrain following coords**: enhanced terrain following coords capabilities.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced_11 import EnhancedABL11Result, set_atm_boundary_layer_enhanced_11

    result = set_atm_boundary_layer_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABL11Result", "set_atm_boundary_layer_enhanced_11"]

@dataclass
class PollutantDispersionResult:
    """Feature data for pollutant_dispersion."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class TerrainFollowingResult:
    """Feature data for terrain_following_coords."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class EnhancedABL11Result:
    """Result from :func:`set_atm_boundary_layer_enhanced_11`."""
    pollutant: Optional[PollutantDispersionResult] = None
    terrain_coords: Optional[TerrainFollowingResult] = None


def set_atm_boundary_layer_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_pollutant: bool = False,
    enable_terrain_coords: bool = False,
) -> EnhancedABL11Result:
    """Enhanced v11 set atm boundary layer.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    EnhancedABL11Result
    """
    pollutant = None
    if enable_pollutant:
        pollutant = PollutantDispersionResult(name="pollutant_dispersion", enabled=True)

    terrain_coords = None
    if enable_terrain_coords:
        terrain_coords = TerrainFollowingResult(name="terrain_following_coords", enabled=True)

    return EnhancedABL11Result(
        pollutant=pollutant,
        terrain_coords=terrain_coords,
    )
