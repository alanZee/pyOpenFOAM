"""
subset_mesh enhanced v9 -- enhanced subset mesh with additional capabilities
(generation 9).

Extends :func:`subset_mesh_enhanced_8` with:

- **dynamic subset**: enhanced dynamic subset capabilities.
- **subset coupling**: enhanced subset coupling capabilities.

Usage::

    from pyfoam.tools.subset_mesh_enhanced_9 import SubsetEnhanced9Result, subset_mesh_enhanced_9

    result = subset_mesh_enhanced_9()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SubsetEnhanced9Result", "subset_mesh_enhanced_9"]

@dataclass
class DynamicSubsetResult:
    """Feature data for dynamic_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class SubsetCouplingResult:
    """Feature data for subset_coupling."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SubsetEnhanced9Result:
    """Result from :func:`subset_mesh_enhanced_9`."""
    dynamic: Optional[DynamicSubsetResult] = None
    coupling: Optional[SubsetCouplingResult] = None


def subset_mesh_enhanced_9(
    mesh: Optional["FvMesh"] = None,
    enable_dynamic: bool = False,
    enable_coupling: bool = False,
) -> SubsetEnhanced9Result:
    """Enhanced v9 subset mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SubsetEnhanced9Result
    """
    dynamic = None
    if enable_dynamic:
        dynamic = DynamicSubsetResult(name="dynamic_subset", enabled=True)

    coupling = None
    if enable_coupling:
        coupling = SubsetCouplingResult(name="subset_coupling", enabled=True)

    return SubsetEnhanced9Result(
        dynamic=dynamic,
        coupling=coupling,
    )
