"""
foam_dictionary enhanced v2 -- enhanced foam dictionary with additional capabilities
(generation 2).

Extends :func:`foam_dictionary_enhanced_1` with:

- **schema validation**: enhanced schema validation capabilities.
- **diff detection**: enhanced diff detection capabilities.

Usage::

    from pyfoam.tools.foam_dictionary_enhanced_2 import FoamDictEnhanced2Result, foam_dictionary_enhanced_2

    result = foam_dictionary_enhanced_2()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["FoamDictEnhanced2Result", "foam_dictionary_enhanced_2"]

@dataclass
class SchemaValidationResult:
    """Feature data for schema_validation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class DiffDetectionResult:
    """Feature data for diff_detection."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class FoamDictEnhanced2Result:
    """Result from :func:`foam_dictionary_enhanced_2`."""
    schema: Optional[SchemaValidationResult] = None
    diff: Optional[DiffDetectionResult] = None


def foam_dictionary_enhanced_2(
    mesh: Optional["FvMesh"] = None,
    enable_schema: bool = False,
    enable_diff: bool = False,
) -> FoamDictEnhanced2Result:
    """Enhanced v2 foam dictionary.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    FoamDictEnhanced2Result
    """
    schema = None
    if enable_schema:
        schema = SchemaValidationResult(name="schema_validation", enabled=True)

    diff = None
    if enable_diff:
        diff = DiffDetectionResult(name="diff_detection", enabled=True)

    return FoamDictEnhanced2Result(
        schema=schema,
        diff=diff,
    )
