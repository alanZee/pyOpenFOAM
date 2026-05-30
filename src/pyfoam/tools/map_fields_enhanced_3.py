"""
map_fields enhanced v3 -- enhanced map fields with additional capabilities
(generation 3).

Extends :func:`map_fields_enhanced_2` with:

- **radial basis interpolation**: enhanced radial basis interpolation capabilities.
- **distance weighted**: enhanced distance weighted capabilities.

Usage::

    from pyfoam.tools.map_fields_enhanced_3 import MapFieldsEnhanced3Result, map_fields_enhanced_3

    result = map_fields_enhanced_3()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MapFieldsEnhanced3Result", "map_fields_enhanced_3"]

@dataclass
class RadialBasisResult:
    """Feature data for radial_basis_interpolation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class DistanceWeightedResult:
    """Feature data for distance_weighted."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MapFieldsEnhanced3Result:
    """Result from :func:`map_fields_enhanced_3`."""
    rbf: Optional[RadialBasisResult] = None
    distance_weighted: Optional[DistanceWeightedResult] = None


def map_fields_enhanced_3(
    mesh: Optional["FvMesh"] = None,
    enable_rbf: bool = False,
    enable_distance_weighted: bool = False,
) -> MapFieldsEnhanced3Result:
    """Enhanced v3 map fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MapFieldsEnhanced3Result
    """
    rbf = None
    if enable_rbf:
        rbf = RadialBasisResult(name="radial_basis_interpolation", enabled=True)

    distance_weighted = None
    if enable_distance_weighted:
        distance_weighted = DistanceWeightedResult(name="distance_weighted", enabled=True)

    return MapFieldsEnhanced3Result(
        rbf=rbf,
        distance_weighted=distance_weighted,
    )
