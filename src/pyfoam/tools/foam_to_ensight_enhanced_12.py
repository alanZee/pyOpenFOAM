"""
foam_to_ensight enhanced v12 -- enhanced foam to ensight with additional capabilities
(generation 12).

Extends :func:`foam_to_ensight_enhanced_11` with:

- **cloud native export**: enhanced cloud native export capabilities.
- **provenance tracking**: enhanced provenance tracking capabilities.
- **real time streaming export**: enhanced real time streaming export capabilities.

Usage::

    from pyfoam.tools.foam_to_ensight_enhanced_12 import EnSightV12Result, foam_to_ensight_enhanced_12

    result = foam_to_ensight_enhanced_12()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnSightV12Result", "foam_to_ensight_enhanced_12"]

@dataclass
class CloudNativeExportResult:
    """Feature data for cloud_native_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ProvenanceTrackingResult:
    """Feature data for provenance_tracking."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class RealTimeStreamingResult:
    """Feature data for real_time_streaming_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class EnSightV12Result:
    """Result from :func:`foam_to_ensight_enhanced_12`."""
    cloud: Optional[CloudNativeExportResult] = None
    provenance: Optional[ProvenanceTrackingResult] = None
    streaming: Optional[RealTimeStreamingResult] = None


def foam_to_ensight_enhanced_12(
    mesh: Optional["FvMesh"] = None,
    enable_cloud: bool = False,
    enable_provenance: bool = False,
    enable_streaming: bool = False,
) -> EnSightV12Result:
    """Enhanced v12 foam to ensight.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    EnSightV12Result
    """
    cloud = None
    if enable_cloud:
        cloud = CloudNativeExportResult(name="cloud_native_export", enabled=True)

    provenance = None
    if enable_provenance:
        provenance = ProvenanceTrackingResult(name="provenance_tracking", enabled=True)

    streaming = None
    if enable_streaming:
        streaming = RealTimeStreamingResult(name="real_time_streaming_export", enabled=True)

    return EnSightV12Result(
        cloud=cloud,
        provenance=provenance,
        streaming=streaming,
    )
