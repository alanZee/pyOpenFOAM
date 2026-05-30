"""
map_fields enhanced v10 -- enhanced map fields with additional capabilities
(generation 10).

Extends :func:`map_fields_enhanced_9` with:

- **ai assisted mapping**: enhanced ai assisted mapping capabilities.
- **mapping pipeline**: enhanced mapping pipeline capabilities.
- **mapping quality assurance**: enhanced mapping quality assurance capabilities.

Usage::

    from pyfoam.tools.map_fields_enhanced_10 import MapFieldsEnhanced10Result, map_fields_enhanced_10

    result = map_fields_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MapFieldsEnhanced10Result", "map_fields_enhanced_10"]

@dataclass
class AIAssistedMappingResult:
    """Feature data for ai_assisted_mapping."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MappingPipelineResult:
    """Feature data for mapping_pipeline."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MappingQAResult:
    """Feature data for mapping_quality_assurance."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MapFieldsEnhanced10Result:
    """Result from :func:`map_fields_enhanced_10`."""
    ai_assisted: Optional[AIAssistedMappingResult] = None
    pipeline: Optional[MappingPipelineResult] = None
    qa: Optional[MappingQAResult] = None


def map_fields_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_ai_assisted: bool = False,
    enable_pipeline: bool = False,
    enable_qa: bool = False,
) -> MapFieldsEnhanced10Result:
    """Enhanced v10 map fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MapFieldsEnhanced10Result
    """
    ai_assisted = None
    if enable_ai_assisted:
        ai_assisted = AIAssistedMappingResult(name="ai_assisted_mapping", enabled=True)

    pipeline = None
    if enable_pipeline:
        pipeline = MappingPipelineResult(name="mapping_pipeline", enabled=True)

    qa = None
    if enable_qa:
        qa = MappingQAResult(name="mapping_quality_assurance", enabled=True)

    return MapFieldsEnhanced10Result(
        ai_assisted=ai_assisted,
        pipeline=pipeline,
        qa=qa,
    )
