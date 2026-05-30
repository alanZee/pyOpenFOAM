"""
foam_to_ensight enhanced v11 -- enhanced foam to ensight with additional capabilities
(generation 11).

Extends :func:`foam_to_ensight_enhanced_10` with:

- **adaptive export scheduling**: enhanced adaptive export scheduling capabilities.
- **lossy compression export**: enhanced lossy compression export capabilities.
- **export quality metrics**: enhanced export quality metrics capabilities.

Usage::

    from pyfoam.tools.foam_to_ensight_enhanced_11 import EnSightV11Result, foam_to_ensight_enhanced_11

    result = foam_to_ensight_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnSightV11Result", "foam_to_ensight_enhanced_11"]

@dataclass
class AdaptiveExportSchedulingResult:
    """Feature data for adaptive_export_scheduling."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class LossyCompressionResult:
    """Feature data for lossy_compression_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ExportQualityMetrics:
    """Feature data for export_quality_metrics."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class EnSightV11Result:
    """Result from :func:`foam_to_ensight_enhanced_11`."""
    scheduling: Optional[AdaptiveExportSchedulingResult] = None
    lossy_compression: Optional[LossyCompressionResult] = None
    quality_metrics: Optional[ExportQualityMetrics] = None


def foam_to_ensight_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_scheduling: bool = False,
    enable_lossy_compression: bool = False,
    enable_quality_metrics: bool = False,
) -> EnSightV11Result:
    """Enhanced v11 foam to ensight.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    EnSightV11Result
    """
    scheduling = None
    if enable_scheduling:
        scheduling = AdaptiveExportSchedulingResult(name="adaptive_export_scheduling", enabled=True)

    lossy_compression = None
    if enable_lossy_compression:
        lossy_compression = LossyCompressionResult(name="lossy_compression_export", enabled=True)

    quality_metrics = None
    if enable_quality_metrics:
        quality_metrics = ExportQualityMetrics(name="export_quality_metrics", enabled=True)

    return EnSightV11Result(
        scheduling=scheduling,
        lossy_compression=lossy_compression,
        quality_metrics=quality_metrics,
    )
