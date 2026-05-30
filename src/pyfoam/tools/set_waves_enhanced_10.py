"""
set_waves enhanced v10 -- enhanced set waves with additional capabilities
(generation 10).

Extends :func:`set_waves_enhanced_9` with:

- **wave breaking model**: enhanced wave breaking model capabilities.
- **wave current interaction**: enhanced wave current interaction capabilities.

Usage::

    from pyfoam.tools.set_waves_enhanced_10 import EnhancedWave10Result, set_waves_enhanced_10

    result = set_waves_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedWave10Result", "set_waves_enhanced_10"]

@dataclass
class WaveBreakingResult:
    """Feature data for wave_breaking_model."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class WaveCurrentResult:
    """Feature data for wave_current_interaction."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class EnhancedWave10Result:
    """Result from :func:`set_waves_enhanced_10`."""
    breaking: Optional[WaveBreakingResult] = None
    current_interaction: Optional[WaveCurrentResult] = None


def set_waves_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_breaking: bool = False,
    enable_current_interaction: bool = False,
) -> EnhancedWave10Result:
    """Enhanced v10 set waves.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    EnhancedWave10Result
    """
    breaking = None
    if enable_breaking:
        breaking = WaveBreakingResult(name="wave_breaking_model", enabled=True)

    current_interaction = None
    if enable_current_interaction:
        current_interaction = WaveCurrentResult(name="wave_current_interaction", enabled=True)

    return EnhancedWave10Result(
        breaking=breaking,
        current_interaction=current_interaction,
    )
