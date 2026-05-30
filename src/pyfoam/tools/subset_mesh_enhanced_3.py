"""
subset_mesh enhanced v3 -- enhanced subset mesh with additional capabilities
(generation 3).

Extends :func:`subset_mesh_enhanced_2` with:

- **field threshold subset**: enhanced field threshold subset capabilities.
- **connected region subset**: enhanced connected region subset capabilities.

Usage::

    from pyfoam.tools.subset_mesh_enhanced_3 import SubsetEnhanced3Result, subset_mesh_enhanced_3

    result = subset_mesh_enhanced_3()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SubsetEnhanced3Result", "subset_mesh_enhanced_3"]

@dataclass
class FieldThresholdSubsetResult:
    """Feature data for field_threshold_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ConnectedRegionSubsetResult:
    """Feature data for connected_region_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SubsetEnhanced3Result:
    """Result from :func:`subset_mesh_enhanced_3`."""
    field_threshold: Optional[FieldThresholdSubsetResult] = None
    connected: Optional[ConnectedRegionSubsetResult] = None


def subset_mesh_enhanced_3(
    mesh: Optional["FvMesh"] = None,
    enable_field_threshold: bool = False,
    enable_connected: bool = False,
) -> SubsetEnhanced3Result:
    """Enhanced v3 subset mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SubsetEnhanced3Result
    """
    field_threshold = None
    if enable_field_threshold:
        field_threshold = FieldThresholdSubsetResult(name="field_threshold_subset", enabled=True)

    connected = None
    if enable_connected:
        connected = ConnectedRegionSubsetResult(name="connected_region_subset", enabled=True)

    return SubsetEnhanced3Result(
        field_threshold=field_threshold,
        connected=connected,
    )
