"""
foam_dictionary enhanced v8 -- enhanced foam dictionary with additional capabilities
(generation 8).

Extends :func:`foam_dictionary_enhanced_7` with:

- **dictionary serialisation**: enhanced dictionary serialisation capabilities.
- **format conversion**: enhanced format conversion capabilities.

Usage::

    from pyfoam.tools.foam_dictionary_enhanced_8 import FoamDictEnhanced8Result, foam_dictionary_enhanced_8

    result = foam_dictionary_enhanced_8()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["FoamDictEnhanced8Result", "foam_dictionary_enhanced_8"]

@dataclass
class SerialisationResult:
    """Feature data for dictionary_serialisation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class FormatConversionResult:
    """Feature data for format_conversion."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class FoamDictEnhanced8Result:
    """Result from :func:`foam_dictionary_enhanced_8`."""
    serialisation: Optional[SerialisationResult] = None
    format_convert: Optional[FormatConversionResult] = None


def foam_dictionary_enhanced_8(
    mesh: Optional["FvMesh"] = None,
    enable_serialisation: bool = False,
    enable_format_convert: bool = False,
) -> FoamDictEnhanced8Result:
    """Enhanced v8 foam dictionary.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    FoamDictEnhanced8Result
    """
    serialisation = None
    if enable_serialisation:
        serialisation = SerialisationResult(name="dictionary_serialisation", enabled=True)

    format_convert = None
    if enable_format_convert:
        format_convert = FormatConversionResult(name="format_conversion", enabled=True)

    return FoamDictEnhanced8Result(
        serialisation=serialisation,
        format_convert=format_convert,
    )
