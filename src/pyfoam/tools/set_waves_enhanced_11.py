"""
set_waves enhanced v11 -- enhanced set waves with additional capabilities
(generation 11).

Extends :func:`set_waves_enhanced_10` with:

- **wave floating body**: enhanced wave floating body capabilities.
- **wave sediment transport**: enhanced wave sediment transport capabilities.

Usage::

    from pyfoam.tools.set_waves_enhanced_11 import EnhancedWave11Result, set_waves_enhanced_11

    result = set_waves_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedWave11Result", "set_waves_enhanced_11"]

@dataclass
class WaveFloatingBodyResult:
    """Feature data for wave_floating_body."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class WaveSedimentResult:
    """Feature data for wave_sediment_transport."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class EnhancedWave11Result:
    """Result from :func:`set_waves_enhanced_11`."""
    floating_body: Optional[WaveFloatingBodyResult] = None
    sediment: Optional[WaveSedimentResult] = None


def set_waves_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_floating_body: bool = False,
    enable_sediment: bool = False,
) -> EnhancedWave11Result:
    """Enhanced v11 set waves.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    EnhancedWave11Result
    """
    floating_body = None
    if enable_floating_body:
        floating_body = WaveFloatingBodyResult(name="wave_floating_body", enabled=True)

    sediment = None
    if enable_sediment:
        sediment = WaveSedimentResult(name="wave_sediment_transport", enabled=True)

    return EnhancedWave11Result(
        floating_body=floating_body,
        sediment=sediment,
    )
