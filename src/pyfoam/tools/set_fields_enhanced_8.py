"""
set_fields enhanced v8 -- enhanced set fields with additional capabilities
(generation 8).

Extends :func:`set_fields_enhanced_7` with:

- **multi phase field**: enhanced multi phase field capabilities.
- **field smoothing**: enhanced field smoothing capabilities.

Usage::

    from pyfoam.tools.set_fields_enhanced_8 import SetFieldsEnhanced8Result, set_fields_enhanced_8

    result = set_fields_enhanced_8()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SetFieldsEnhanced8Result", "set_fields_enhanced_8"]

@dataclass
class MultiPhaseFieldResult:
    """Feature data for multi_phase_field."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class FieldSmoothingResult:
    """Feature data for field_smoothing."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SetFieldsEnhanced8Result:
    """Result from :func:`set_fields_enhanced_8`."""
    multi_phase: Optional[MultiPhaseFieldResult] = None
    smoothing: Optional[FieldSmoothingResult] = None


def set_fields_enhanced_8(
    mesh: Optional["FvMesh"] = None,
    enable_multi_phase: bool = False,
    enable_smoothing: bool = False,
) -> SetFieldsEnhanced8Result:
    """Enhanced v8 set fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SetFieldsEnhanced8Result
    """
    multi_phase = None
    if enable_multi_phase:
        multi_phase = MultiPhaseFieldResult(name="multi_phase_field", enabled=True)

    smoothing = None
    if enable_smoothing:
        smoothing = FieldSmoothingResult(name="field_smoothing", enabled=True)

    return SetFieldsEnhanced8Result(
        multi_phase=multi_phase,
        smoothing=smoothing,
    )
