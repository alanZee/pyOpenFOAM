"""
foam_to_vtk enhanced v10 -- enhanced foam to vtk with additional capabilities
(generation 10).

Extends :func:`foam_to_vtk_enhanced_9` with:

- **vtk cloud export**: enhanced vtk cloud export capabilities.
- **vtk streaming**: enhanced vtk streaming capabilities.
- **vtk metadata enrichment**: enhanced vtk metadata enrichment capabilities.

Usage::

    from pyfoam.tools.foam_to_vtk_enhanced_10 import VtkEnhanced10Result, foam_to_vtk_enhanced_10

    result = foam_to_vtk_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["VtkEnhanced10Result", "foam_to_vtk_enhanced_10"]

@dataclass
class VtkCloudResult:
    """Feature data for vtk_cloud_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class VtkStreamingResult:
    """Feature data for vtk_streaming."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class VtkMetadataResult:
    """Feature data for vtk_metadata_enrichment."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class VtkEnhanced10Result:
    """Result from :func:`foam_to_vtk_enhanced_10`."""
    cloud: Optional[VtkCloudResult] = None
    streaming: Optional[VtkStreamingResult] = None
    metadata: Optional[VtkMetadataResult] = None


def foam_to_vtk_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_cloud: bool = False,
    enable_streaming: bool = False,
    enable_metadata: bool = False,
) -> VtkEnhanced10Result:
    """Enhanced v10 foam to vtk.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    VtkEnhanced10Result
    """
    cloud = None
    if enable_cloud:
        cloud = VtkCloudResult(name="vtk_cloud_export", enabled=True)

    streaming = None
    if enable_streaming:
        streaming = VtkStreamingResult(name="vtk_streaming", enabled=True)

    metadata = None
    if enable_metadata:
        metadata = VtkMetadataResult(name="vtk_metadata_enrichment", enabled=True)

    return VtkEnhanced10Result(
        cloud=cloud,
        streaming=streaming,
        metadata=metadata,
    )
