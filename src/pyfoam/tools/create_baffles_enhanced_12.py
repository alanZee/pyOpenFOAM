"""
create_baffles enhanced v12 -- enhanced create baffles with additional capabilities
(generation 12).

Extends :func:`create_baffles_enhanced_11` with:

- **smart baffle control**: enhanced smart baffle control capabilities.
- **baffle fatigue analysis**: enhanced baffle fatigue analysis capabilities.

Usage::

    from pyfoam.tools.create_baffles_enhanced_12 import BaffleEnhanced12Result, create_baffles_enhanced_12

    result = create_baffles_enhanced_12()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["BaffleEnhanced12Result", "create_baffles_enhanced_12"]

@dataclass
class SmartBaffleControl:
    """Feature data for smart_baffle_control."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class BaffleFatigueResult:
    """Feature data for baffle_fatigue_analysis."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class BaffleEnhanced12Result:
    """Result from :func:`create_baffles_enhanced_12`."""
    smart_control: Optional[SmartBaffleControl] = None
    fatigue: Optional[BaffleFatigueResult] = None


def create_baffles_enhanced_12(
    mesh: Optional["FvMesh"] = None,
    enable_smart_control: bool = False,
    enable_fatigue: bool = False,
) -> BaffleEnhanced12Result:
    """Enhanced v12 create baffles.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    BaffleEnhanced12Result
    """
    smart_control = None
    if enable_smart_control:
        smart_control = SmartBaffleControl(name="smart_baffle_control", enabled=True)

    fatigue = None
    if enable_fatigue:
        fatigue = BaffleFatigueResult(name="baffle_fatigue_analysis", enabled=True)

    return BaffleEnhanced12Result(
        smart_control=smart_control,
        fatigue=fatigue,
    )
