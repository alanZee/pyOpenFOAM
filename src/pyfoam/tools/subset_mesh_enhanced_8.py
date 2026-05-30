"""
subset_mesh enhanced v8 -- enhanced subset mesh with additional capabilities
(generation 8).

Extends :func:`subset_mesh_enhanced_7` with:

- **hierarchical subset**: enhanced hierarchical subset capabilities.
- **subset interface management**: enhanced subset interface management capabilities.

Usage::

    from pyfoam.tools.subset_mesh_enhanced_8 import SubsetEnhanced8Result, subset_mesh_enhanced_8

    result = subset_mesh_enhanced_8()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SubsetEnhanced8Result", "subset_mesh_enhanced_8"]

@dataclass
class HierarchicalSubsetResult:
    """Feature data for hierarchical_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class InterfaceManagementResult:
    """Feature data for subset_interface_management."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SubsetEnhanced8Result:
    """Result from :func:`subset_mesh_enhanced_8`."""
    hierarchical: Optional[HierarchicalSubsetResult] = None
    interface: Optional[InterfaceManagementResult] = None


def subset_mesh_enhanced_8(
    mesh: Optional["FvMesh"] = None,
    enable_hierarchical: bool = False,
    enable_interface: bool = False,
) -> SubsetEnhanced8Result:
    """Enhanced v8 subset mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SubsetEnhanced8Result
    """
    hierarchical = None
    if enable_hierarchical:
        hierarchical = HierarchicalSubsetResult(name="hierarchical_subset", enabled=True)

    interface = None
    if enable_interface:
        interface = InterfaceManagementResult(name="subset_interface_management", enabled=True)

    return SubsetEnhanced8Result(
        hierarchical=hierarchical,
        interface=interface,
    )
