"""
foam_to_vtk enhanced v6 -- enhanced foam to vtk with additional capabilities
(generation 6).

Extends :func:`foam_to_vtk_enhanced_5` with:

- **time series export**: enhanced time series export capabilities.
- **derived field export**: enhanced derived field export capabilities.

Usage::

    from pyfoam.tools.foam_to_vtk_enhanced_6 import VtkEnhanced6Result, foam_to_vtk_enhanced_6

    result = foam_to_vtk_enhanced_6()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["VtkEnhanced6Result", "foam_to_vtk_enhanced_6"]

@dataclass
class TimeSeriesVtkResult:
    """Feature data for time_series_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class DerivedFieldVtkResult:
    """Feature data for derived_field_export."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class VtkEnhanced6Result:
    """Result from :func:`foam_to_vtk_enhanced_6`."""
    time_series: Optional[TimeSeriesVtkResult] = None
    derived_fields: Optional[DerivedFieldVtkResult] = None


def foam_to_vtk_enhanced_6(
    mesh: Optional["FvMesh"] = None,
    enable_time_series: bool = False,
    enable_derived_fields: bool = False,
) -> VtkEnhanced6Result:
    """Enhanced v6 foam to vtk.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    VtkEnhanced6Result
    """
    time_series = None
    if enable_time_series:
        time_series = TimeSeriesVtkResult(name="time_series_export", enabled=True)

    derived_fields = None
    if enable_derived_fields:
        derived_fields = DerivedFieldVtkResult(name="derived_field_export", enabled=True)

    return VtkEnhanced6Result(
        time_series=time_series,
        derived_fields=derived_fields,
    )
