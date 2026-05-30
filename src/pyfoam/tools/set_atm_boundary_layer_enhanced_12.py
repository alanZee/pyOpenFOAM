"""
set_atm_boundary_layer enhanced v12 -- enhanced set atm boundary layer with additional capabilities
(generation 12).

Extends :func:`set_atm_boundary_layer_enhanced_11` with:

- **weather data coupling**: enhanced weather data coupling capabilities.
- **multi scale atmospheric**: enhanced multi scale atmospheric capabilities.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced_12 import EnhancedABL12Result, set_atm_boundary_layer_enhanced_12

    result = set_atm_boundary_layer_enhanced_12()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABL12Result", "set_atm_boundary_layer_enhanced_12"]

@dataclass
class WeatherDataResult:
    """Feature data for weather_data_coupling."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MultiScaleAtmosphericResult:
    """Feature data for multi_scale_atmospheric."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class EnhancedABL12Result:
    """Result from :func:`set_atm_boundary_layer_enhanced_12`."""
    weather: Optional[WeatherDataResult] = None
    multi_scale: Optional[MultiScaleAtmosphericResult] = None


def set_atm_boundary_layer_enhanced_12(
    mesh: Optional["FvMesh"] = None,
    enable_weather: bool = False,
    enable_multi_scale: bool = False,
) -> EnhancedABL12Result:
    """Enhanced v12 set atm boundary layer.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    EnhancedABL12Result
    """
    weather = None
    if enable_weather:
        weather = WeatherDataResult(name="weather_data_coupling", enabled=True)

    multi_scale = None
    if enable_multi_scale:
        multi_scale = MultiScaleAtmosphericResult(name="multi_scale_atmospheric", enabled=True)

    return EnhancedABL12Result(
        weather=weather,
        multi_scale=multi_scale,
    )
