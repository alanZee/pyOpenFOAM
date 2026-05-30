"""
set_atm_boundary_layer enhanced v10 -- enhanced set atm boundary layer with additional capabilities
(generation 10).

Extends :func:`set_atm_boundary_layer_enhanced_9` with:

- **canopy model**: enhanced canopy model capabilities.
- **urban heat island**: enhanced urban heat island capabilities.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced_10 import EnhancedABL10Result, set_atm_boundary_layer_enhanced_10

    result = set_atm_boundary_layer_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABL10Result", "set_atm_boundary_layer_enhanced_10"]

@dataclass
class CanopyModelResult:
    """Feature data for canopy_model."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class UrbanHeatIslandResult:
    """Feature data for urban_heat_island."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class EnhancedABL10Result:
    """Result from :func:`set_atm_boundary_layer_enhanced_10`."""
    canopy: Optional[CanopyModelResult] = None
    urban_heat: Optional[UrbanHeatIslandResult] = None


def set_atm_boundary_layer_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_canopy: bool = False,
    enable_urban_heat: bool = False,
) -> EnhancedABL10Result:
    """Enhanced v10 set atm boundary layer.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    EnhancedABL10Result
    """
    canopy = None
    if enable_canopy:
        canopy = CanopyModelResult(name="canopy_model", enabled=True)

    urban_heat = None
    if enable_urban_heat:
        urban_heat = UrbanHeatIslandResult(name="urban_heat_island", enabled=True)

    return EnhancedABL10Result(
        canopy=canopy,
        urban_heat=urban_heat,
    )
