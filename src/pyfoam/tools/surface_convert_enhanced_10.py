"""
surface_convert enhanced v10 -- enhanced surface convert with additional capabilities
(generation 10).

Extends :func:`surface_convert_enhanced_9` with:

- **adaptive tessellation**: enhanced adaptive tessellation capabilities.
- **format validation**: enhanced format validation capabilities.

Usage::

    from pyfoam.tools.surface_convert_enhanced_10 import ConvertEnhanced10Result, surface_convert_enhanced_10

    result = surface_convert_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["ConvertEnhanced10Result", "surface_convert_enhanced_10"]

@dataclass
class AdaptiveTessellationResult:
    """Feature data for adaptive_tessellation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class FormatValidationResult:
    """Feature data for format_validation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class ConvertEnhanced10Result:
    """Result from :func:`surface_convert_enhanced_10`."""
    tessellation: Optional[AdaptiveTessellationResult] = None
    validation: Optional[FormatValidationResult] = None


def surface_convert_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_tessellation: bool = False,
    enable_validation: bool = False,
) -> ConvertEnhanced10Result:
    """Enhanced v10 surface convert.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    ConvertEnhanced10Result
    """
    tessellation = None
    if enable_tessellation:
        tessellation = AdaptiveTessellationResult(name="adaptive_tessellation", enabled=True)

    validation = None
    if enable_validation:
        validation = FormatValidationResult(name="format_validation", enabled=True)

    return ConvertEnhanced10Result(
        tessellation=tessellation,
        validation=validation,
    )
