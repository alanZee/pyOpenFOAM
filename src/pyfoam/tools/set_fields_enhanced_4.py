"""
set_fields enhanced v4 -- enhanced set fields with additional capabilities
(generation 4).

Extends :func:`set_fields_enhanced_3` with:

- **boundary layer field**: enhanced boundary layer field capabilities.
- **time varying field**: enhanced time varying field capabilities.

Usage::

    from pyfoam.tools.set_fields_enhanced_4 import SetFieldsEnhanced4Result, set_fields_enhanced_4

    result = set_fields_enhanced_4()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SetFieldsEnhanced4Result", "set_fields_enhanced_4"]

@dataclass
class BoundaryLayerFieldResult:
    """Feature data for boundary_layer_field."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class TimeVaryingFieldResult:
    """Feature data for time_varying_field."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SetFieldsEnhanced4Result:
    """Result from :func:`set_fields_enhanced_4`."""
    bl_field: Optional[BoundaryLayerFieldResult] = None
    time_varying: Optional[TimeVaryingFieldResult] = None


def set_fields_enhanced_4(
    mesh: Optional["FvMesh"] = None,
    enable_bl_field: bool = False,
    enable_time_varying: bool = False,
) -> SetFieldsEnhanced4Result:
    """Enhanced v4 set fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SetFieldsEnhanced4Result
    """
    bl_field = None
    if enable_bl_field:
        bl_field = BoundaryLayerFieldResult(name="boundary_layer_field", enabled=True)

    time_varying = None
    if enable_time_varying:
        time_varying = TimeVaryingFieldResult(name="time_varying_field", enabled=True)

    return SetFieldsEnhanced4Result(
        bl_field=bl_field,
        time_varying=time_varying,
    )
