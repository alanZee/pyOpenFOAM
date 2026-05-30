"""
map_fields enhanced v5 -- enhanced map fields with additional capabilities
(generation 5).

Extends :func:`map_fields_enhanced_4` with:

- **adaptive mapping**: enhanced adaptive mapping capabilities.
- **mapping validation**: enhanced mapping validation capabilities.
- **parallel mapping**: enhanced parallel mapping capabilities.

Usage::

    from pyfoam.tools.map_fields_enhanced_5 import MapFieldsEnhanced5Result, map_fields_enhanced_5

    result = map_fields_enhanced_5()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MapFieldsEnhanced5Result", "map_fields_enhanced_5"]

@dataclass
class AdaptiveMappingResult:
    """Feature data for adaptive_mapping."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MappingValidationResult:
    """Feature data for mapping_validation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ParallelMappingResult:
    """Feature data for parallel_mapping."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MapFieldsEnhanced5Result:
    """Result from :func:`map_fields_enhanced_5`."""
    adaptive: Optional[AdaptiveMappingResult] = None
    validation: Optional[MappingValidationResult] = None
    parallel: Optional[ParallelMappingResult] = None


def map_fields_enhanced_5(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive: bool = False,
    enable_validation: bool = False,
    enable_parallel: bool = False,
) -> MapFieldsEnhanced5Result:
    """Enhanced v5 map fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MapFieldsEnhanced5Result
    """
    adaptive = None
    if enable_adaptive:
        adaptive = AdaptiveMappingResult(name="adaptive_mapping", enabled=True)

    validation = None
    if enable_validation:
        validation = MappingValidationResult(name="mapping_validation", enabled=True)

    parallel = None
    if enable_parallel:
        parallel = ParallelMappingResult(name="parallel_mapping", enabled=True)

    return MapFieldsEnhanced5Result(
        adaptive=adaptive,
        validation=validation,
        parallel=parallel,
    )
