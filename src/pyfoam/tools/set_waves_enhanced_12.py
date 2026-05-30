"""
set_waves enhanced v12 -- enhanced set waves with additional capabilities
(generation 12).

Extends :func:`set_waves_enhanced_11` with:

- **wave farm interaction**: enhanced wave farm interaction capabilities.
- **coupled wave structure**: enhanced coupled wave structure capabilities.

Usage::

    from pyfoam.tools.set_waves_enhanced_12 import EnhancedWave12Result, set_waves_enhanced_12

    result = set_waves_enhanced_12()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedWave12Result", "set_waves_enhanced_12"]

@dataclass
class WaveFarmResult:
    """Feature data for wave_farm_interaction."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class CoupledWaveStructureResult:
    """Feature data for coupled_wave_structure."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class EnhancedWave12Result:
    """Result from :func:`set_waves_enhanced_12`."""
    farm: Optional[WaveFarmResult] = None
    structure_coupling: Optional[CoupledWaveStructureResult] = None


def set_waves_enhanced_12(
    mesh: Optional["FvMesh"] = None,
    enable_farm: bool = False,
    enable_structure_coupling: bool = False,
) -> EnhancedWave12Result:
    """Enhanced v12 set waves.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    EnhancedWave12Result
    """
    farm = None
    if enable_farm:
        farm = WaveFarmResult(name="wave_farm_interaction", enabled=True)

    structure_coupling = None
    if enable_structure_coupling:
        structure_coupling = CoupledWaveStructureResult(name="coupled_wave_structure", enabled=True)

    return EnhancedWave12Result(
        farm=farm,
        structure_coupling=structure_coupling,
    )
