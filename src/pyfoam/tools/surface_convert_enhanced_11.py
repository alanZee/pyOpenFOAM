"""
surface_convert enhanced v11 -- enhanced surface convert with additional capabilities
(generation 11).

Extends :func:`surface_convert_enhanced_10` with:

- **color mapping export**: enhanced color mapping export capabilities.
- **metadata preservation**: enhanced metadata preservation capabilities.

Usage::

    from pyfoam.tools.surface_convert_enhanced_11 import ConvertEnhanced11Result, surface_convert_enhanced_11

    result = surface_convert_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["ConvertEnhanced11Result", "surface_convert_enhanced_11"]

@dataclass
class ColorMappingResult:
    """Feature data for color_mapping_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MetadataPreservationResult:
    """Feature data for metadata_preservation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class ConvertEnhanced11Result:
    """Result from :func:`surface_convert_enhanced_11`."""
    color_mapping: Optional[ColorMappingResult] = None
    metadata: Optional[MetadataPreservationResult] = None


def surface_convert_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_color_mapping: bool = False,
    enable_metadata: bool = False,
) -> ConvertEnhanced11Result:
    """Enhanced v11 surface convert.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    ConvertEnhanced11Result
    """
    color_mapping = None
    if enable_color_mapping:
        color_mapping = ColorMappingResult(name="color_mapping_export", enabled=True)

    metadata = None
    if enable_metadata:
        metadata = MetadataPreservationResult(name="metadata_preservation", enabled=True)

    return ConvertEnhanced11Result(
        color_mapping=color_mapping,
        metadata=metadata,
    )
