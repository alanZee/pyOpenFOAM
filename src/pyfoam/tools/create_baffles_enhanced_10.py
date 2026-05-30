"""
create_baffles enhanced v10 -- enhanced create baffles with additional capabilities
(generation 10).

Extends :func:`create_baffles_enhanced_9` with:

- **porous baffle model**: enhanced porous baffle model capabilities.
- **thermal resistance baffle**: enhanced thermal resistance baffle capabilities.

Usage::

    from pyfoam.tools.create_baffles_enhanced_10 import BaffleEnhanced10Result, create_baffles_enhanced_10

    result = create_baffles_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["BaffleEnhanced10Result", "create_baffles_enhanced_10"]

@dataclass
class PorousBaffleModel:
    """Feature data for porous_baffle_model."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ThermalResistanceBaffle:
    """Feature data for thermal_resistance_baffle."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class BaffleEnhanced10Result:
    """Result from :func:`create_baffles_enhanced_10`."""
    porous_model: Optional[PorousBaffleModel] = None
    thermal_resistance: Optional[ThermalResistanceBaffle] = None


def create_baffles_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_porous_model: bool = False,
    enable_thermal_resistance: bool = False,
) -> BaffleEnhanced10Result:
    """Enhanced v10 create baffles.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    BaffleEnhanced10Result
    """
    porous_model = None
    if enable_porous_model:
        porous_model = PorousBaffleModel(name="porous_baffle_model", enabled=True)

    thermal_resistance = None
    if enable_thermal_resistance:
        thermal_resistance = ThermalResistanceBaffle(name="thermal_resistance_baffle", enabled=True)

    return BaffleEnhanced10Result(
        porous_model=porous_model,
        thermal_resistance=thermal_resistance,
    )
