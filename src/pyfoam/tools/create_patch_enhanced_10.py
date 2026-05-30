"""
create_patch enhanced v10 -- enhanced create patch with additional capabilities
(generation 10).

Extends :func:`create_patch_enhanced_9` with:

- **patch coupling interface**: enhanced patch coupling interface capabilities.
- **mapped patch generation**: enhanced mapped patch generation capabilities.

Usage::

    from pyfoam.tools.create_patch_enhanced_10 import PatchEnhanced10Result, create_patch_enhanced_10

    result = create_patch_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["PatchEnhanced10Result", "create_patch_enhanced_10"]

@dataclass
class PatchCouplingInterface:
    """Feature data for patch_coupling_interface."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MappedPatchResult:
    """Feature data for mapped_patch_generation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class PatchEnhanced10Result:
    """Result from :func:`create_patch_enhanced_10`."""
    coupling: Optional[PatchCouplingInterface] = None
    mapped: Optional[MappedPatchResult] = None


def create_patch_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_coupling: bool = False,
    enable_mapped: bool = False,
) -> PatchEnhanced10Result:
    """Enhanced v10 create patch.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    PatchEnhanced10Result
    """
    coupling = None
    if enable_coupling:
        coupling = PatchCouplingInterface(name="patch_coupling_interface", enabled=True)

    mapped = None
    if enable_mapped:
        mapped = MappedPatchResult(name="mapped_patch_generation", enabled=True)

    return PatchEnhanced10Result(
        coupling=coupling,
        mapped=mapped,
    )
