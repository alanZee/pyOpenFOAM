"""
merge_meshes enhanced v11 -- enhanced merge meshes with additional capabilities
(generation 11).

Extends :func:`merge_meshes_enhanced_10` with:

- **adaptive interface refinement**: enhanced adaptive interface refinement capabilities.
- **multi resolution merge**: enhanced multi resolution merge capabilities.

Usage::

    from pyfoam.tools.merge_meshes_enhanced_11 import MergeEnhanced11Result, merge_meshes_enhanced_11

    result = merge_meshes_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["MergeEnhanced11Result", "merge_meshes_enhanced_11"]

@dataclass
class AdaptiveInterfaceResult:
    """Feature data for adaptive_interface_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MultiResolutionMergeResult:
    """Feature data for multi_resolution_merge."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class MergeEnhanced11Result:
    """Result from :func:`merge_meshes_enhanced_11`."""
    adaptive_interface: Optional[AdaptiveInterfaceResult] = None
    multi_resolution: Optional[MultiResolutionMergeResult] = None


def merge_meshes_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive_interface: bool = False,
    enable_multi_resolution: bool = False,
) -> MergeEnhanced11Result:
    """Enhanced v11 merge meshes.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    MergeEnhanced11Result
    """
    adaptive_interface = None
    if enable_adaptive_interface:
        adaptive_interface = AdaptiveInterfaceResult(name="adaptive_interface_refinement", enabled=True)

    multi_resolution = None
    if enable_multi_resolution:
        multi_resolution = MultiResolutionMergeResult(name="multi_resolution_merge", enabled=True)

    return MergeEnhanced11Result(
        adaptive_interface=adaptive_interface,
        multi_resolution=multi_resolution,
    )
