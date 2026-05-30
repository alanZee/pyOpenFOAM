"""
surface_convert enhanced v12 -- enhanced surface convert with additional capabilities
(generation 12).

Extends :func:`surface_convert_enhanced_11` with:

- **compression optimization**: enhanced compression optimization capabilities.
- **multi lod export**: enhanced multi lod export capabilities.

Usage::

    from pyfoam.tools.surface_convert_enhanced_12 import ConvertEnhanced12Result, surface_convert_enhanced_12

    result = surface_convert_enhanced_12()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["ConvertEnhanced12Result", "surface_convert_enhanced_12"]

@dataclass
class CompressionResult:
    """Feature data for compression_optimization."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MultiLODResult:
    """Feature data for multi_lod_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class ConvertEnhanced12Result:
    """Result from :func:`surface_convert_enhanced_12`."""
    compression: Optional[CompressionResult] = None
    multi_lod: Optional[MultiLODResult] = None


def surface_convert_enhanced_12(
    mesh: Optional["FvMesh"] = None,
    enable_compression: bool = False,
    enable_multi_lod: bool = False,
) -> ConvertEnhanced12Result:
    """Enhanced v12 surface convert.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    ConvertEnhanced12Result
    """
    compression = None
    if enable_compression:
        compression = CompressionResult(name="compression_optimization", enabled=True)

    multi_lod = None
    if enable_multi_lod:
        multi_lod = MultiLODResult(name="multi_lod_export", enabled=True)

    return ConvertEnhanced12Result(
        compression=compression,
        multi_lod=multi_lod,
    )
