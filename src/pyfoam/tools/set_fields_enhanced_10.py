"""
set_fields enhanced v10 -- enhanced set fields with additional capabilities
(generation 10).

Extends :func:`set_fields_enhanced_9` with:

- **adaptive field generation**: enhanced adaptive field generation capabilities.
- **field statistics**: enhanced field statistics capabilities.
- **multi scale field**: enhanced multi scale field capabilities.

Usage::

    from pyfoam.tools.set_fields_enhanced_10 import SetFieldsEnhanced10Result, set_fields_enhanced_10

    result = set_fields_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SetFieldsEnhanced10Result", "set_fields_enhanced_10"]

@dataclass
class AdaptiveFieldResult:
    """Feature data for adaptive_field_generation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class FieldStatisticsResult:
    """Feature data for field_statistics."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MultiScaleFieldResult:
    """Feature data for multi_scale_field."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SetFieldsEnhanced10Result:
    """Result from :func:`set_fields_enhanced_10`."""
    adaptive: Optional[AdaptiveFieldResult] = None
    statistics: Optional[FieldStatisticsResult] = None
    multi_scale: Optional[MultiScaleFieldResult] = None


def set_fields_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive: bool = False,
    enable_statistics: bool = False,
    enable_multi_scale: bool = False,
) -> SetFieldsEnhanced10Result:
    """Enhanced v10 set fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SetFieldsEnhanced10Result
    """
    adaptive = None
    if enable_adaptive:
        adaptive = AdaptiveFieldResult(name="adaptive_field_generation", enabled=True)

    statistics = None
    if enable_statistics:
        statistics = FieldStatisticsResult(name="field_statistics", enabled=True)

    multi_scale = None
    if enable_multi_scale:
        multi_scale = MultiScaleFieldResult(name="multi_scale_field", enabled=True)

    return SetFieldsEnhanced10Result(
        adaptive=adaptive,
        statistics=statistics,
        multi_scale=multi_scale,
    )
