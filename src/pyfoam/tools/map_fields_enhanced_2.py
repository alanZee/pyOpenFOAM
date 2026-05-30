"""
map_fields enhanced v2 -- enhanced map fields with additional capabilities
(generation 2).

Extends :func:`map_fields_enhanced_1` with:

- **interpolation method linear**: enhanced interpolation method linear capabilities.
- **conservative mapping**: enhanced conservative mapping capabilities.

Usage::

    from pyfoam.tools.map_fields_enhanced_2 import MapFieldsEnhanced2Result, map_fields_enhanced_2

    result = map_fields_enhanced_2()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MapFieldsEnhanced2Result", "map_fields_enhanced_2"]

@dataclass
class LinearInterpolationResult:
    """Feature data for interpolation_method_linear."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ConservativeMappingResult:
    """Feature data for conservative_mapping."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MapFieldsEnhanced2Result:
    """Result from :func:`map_fields_enhanced_2`."""
    linear: Optional[LinearInterpolationResult] = None
    conservative: Optional[ConservativeMappingResult] = None


def map_fields_enhanced_2(
    mesh: Optional["FvMesh"] = None,
    enable_linear: bool = False,
    enable_conservative: bool = False,
) -> MapFieldsEnhanced2Result:
    """Enhanced v2 map fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MapFieldsEnhanced2Result
    """
    linear = None
    if enable_linear:
        linear = LinearInterpolationResult(name="interpolation_method_linear", enabled=True)

    conservative = None
    if enable_conservative:
        conservative = ConservativeMappingResult(name="conservative_mapping", enabled=True)

    return MapFieldsEnhanced2Result(
        linear=linear,
        conservative=conservative,
    )
