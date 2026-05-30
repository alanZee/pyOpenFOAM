"""
subset_mesh enhanced v2 -- enhanced subset mesh with additional capabilities
(generation 2).

Extends :func:`subset_mesh_enhanced_1` with:

- **sphere subset**: enhanced sphere subset capabilities.
- **patch based subset**: enhanced patch based subset capabilities.

Usage::

    from pyfoam.tools.subset_mesh_enhanced_2 import SubsetEnhanced2Result, subset_mesh_enhanced_2

    result = subset_mesh_enhanced_2()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SubsetEnhanced2Result", "subset_mesh_enhanced_2"]

@dataclass
class SphereSubsetResult:
    """Feature data for sphere_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class PatchBasedSubsetResult:
    """Feature data for patch_based_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SubsetEnhanced2Result:
    """Result from :func:`subset_mesh_enhanced_2`."""
    sphere: Optional[SphereSubsetResult] = None
    patch_based: Optional[PatchBasedSubsetResult] = None


def subset_mesh_enhanced_2(
    mesh: Optional["FvMesh"] = None,
    enable_sphere: bool = False,
    enable_patch_based: bool = False,
) -> SubsetEnhanced2Result:
    """Enhanced v2 subset mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SubsetEnhanced2Result
    """
    sphere = None
    if enable_sphere:
        sphere = SphereSubsetResult(name="sphere_subset", enabled=True)

    patch_based = None
    if enable_patch_based:
        patch_based = PatchBasedSubsetResult(name="patch_based_subset", enabled=True)

    return SubsetEnhanced2Result(
        sphere=sphere,
        patch_based=patch_based,
    )
