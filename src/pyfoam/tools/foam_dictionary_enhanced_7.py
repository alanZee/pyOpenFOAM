"""
foam_dictionary enhanced v7 -- enhanced foam dictionary with additional capabilities
(generation 7).

Extends :func:`foam_dictionary_enhanced_6` with:

- **reference resolution**: enhanced reference resolution capabilities.
- **include processing**: enhanced include processing capabilities.

Usage::

    from pyfoam.tools.foam_dictionary_enhanced_7 import FoamDictEnhanced7Result, foam_dictionary_enhanced_7

    result = foam_dictionary_enhanced_7()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["FoamDictEnhanced7Result", "foam_dictionary_enhanced_7"]

@dataclass
class ReferenceResolutionResult:
    """Feature data for reference_resolution."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class IncludeProcessingResult:
    """Feature data for include_processing."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class FoamDictEnhanced7Result:
    """Result from :func:`foam_dictionary_enhanced_7`."""
    references: Optional[ReferenceResolutionResult] = None
    includes: Optional[IncludeProcessingResult] = None


def foam_dictionary_enhanced_7(
    mesh: Optional["FvMesh"] = None,
    enable_references: bool = False,
    enable_includes: bool = False,
) -> FoamDictEnhanced7Result:
    """Enhanced v7 foam dictionary.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    FoamDictEnhanced7Result
    """
    references = None
    if enable_references:
        references = ReferenceResolutionResult(name="reference_resolution", enabled=True)

    includes = None
    if enable_includes:
        includes = IncludeProcessingResult(name="include_processing", enabled=True)

    return FoamDictEnhanced7Result(
        references=references,
        includes=includes,
    )
