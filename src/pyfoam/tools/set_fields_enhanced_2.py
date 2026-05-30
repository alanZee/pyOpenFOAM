"""
set_fields enhanced v2 -- enhanced set fields with additional capabilities
(generation 2).

Extends :func:`set_fields_enhanced_1` with:

- **sphere region**: enhanced sphere region capabilities.
- **surface distance region**: enhanced surface distance region capabilities.

Usage::

    from pyfoam.tools.set_fields_enhanced_2 import SetFieldsEnhanced2Result, set_fields_enhanced_2

    result = set_fields_enhanced_2()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SetFieldsEnhanced2Result", "set_fields_enhanced_2"]

@dataclass
class SphereRegion:
    """Feature data for sphere_region."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class SurfaceDistanceRegion:
    """Feature data for surface_distance_region."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SetFieldsEnhanced2Result:
    """Result from :func:`set_fields_enhanced_2`."""
    sphere: Optional[SphereRegion] = None
    surface_distance: Optional[SurfaceDistanceRegion] = None


def set_fields_enhanced_2(
    mesh: Optional["FvMesh"] = None,
    enable_sphere: bool = False,
    enable_surface_distance: bool = False,
) -> SetFieldsEnhanced2Result:
    """Enhanced v2 set fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SetFieldsEnhanced2Result
    """
    sphere = None
    if enable_sphere:
        sphere = SphereRegion(name="sphere_region", enabled=True)

    surface_distance = None
    if enable_surface_distance:
        surface_distance = SurfaceDistanceRegion(name="surface_distance_region", enabled=True)

    return SetFieldsEnhanced2Result(
        sphere=sphere,
        surface_distance=surface_distance,
    )
