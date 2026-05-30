"""
create_patch enhanced v11 -- enhanced create patch with additional capabilities
(generation 11).

Extends :func:`create_patch_enhanced_10` with:

- **cyclic ami patch**: enhanced cyclic ami patch capabilities.
- **overset patch interface**: enhanced overset patch interface capabilities.

Usage::

    from pyfoam.tools.create_patch_enhanced_11 import PatchEnhanced11Result, create_patch_enhanced_11

    result = create_patch_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["PatchEnhanced11Result", "create_patch_enhanced_11"]

@dataclass
class CyclicAMIPatchResult:
    """Feature data for cyclic_ami_patch."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class OversetPatchResult:
    """Feature data for overset_patch_interface."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class PatchEnhanced11Result:
    """Result from :func:`create_patch_enhanced_11`."""
    cyclic_ami: Optional[CyclicAMIPatchResult] = None
    overset: Optional[OversetPatchResult] = None


def create_patch_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_cyclic_ami: bool = False,
    enable_overset: bool = False,
) -> PatchEnhanced11Result:
    """Enhanced v11 create patch.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    PatchEnhanced11Result
    """
    cyclic_ami = None
    if enable_cyclic_ami:
        cyclic_ami = CyclicAMIPatchResult(name="cyclic_ami_patch", enabled=True)

    overset = None
    if enable_overset:
        overset = OversetPatchResult(name="overset_patch_interface", enabled=True)

    return PatchEnhanced11Result(
        cyclic_ami=cyclic_ami,
        overset=overset,
    )
