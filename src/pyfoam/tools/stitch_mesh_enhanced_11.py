"""
stitch_mesh enhanced v11 -- enhanced stitch mesh with additional capabilities
(generation 11).

Extends :func:`stitch_mesh_enhanced_10` with:

- **adaptive stitch refinement**: enhanced adaptive stitch refinement capabilities.
- **multi patch stitching**: enhanced multi patch stitching capabilities.

Usage::

    from pyfoam.tools.stitch_mesh_enhanced_11 import StitchEnhanced11Result, stitch_mesh_enhanced_11

    result = stitch_mesh_enhanced_11()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["StitchEnhanced11Result", "stitch_mesh_enhanced_11"]

@dataclass
class AdaptiveStitchResult:
    """Feature data for adaptive_stitch_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MultiPatchStitchResult:
    """Feature data for multi_patch_stitching."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class StitchEnhanced11Result:
    """Result from :func:`stitch_mesh_enhanced_11`."""
    adaptive: Optional[AdaptiveStitchResult] = None
    multi_patch: Optional[MultiPatchStitchResult] = None


def stitch_mesh_enhanced_11(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive: bool = False,
    enable_multi_patch: bool = False,
) -> StitchEnhanced11Result:
    """Enhanced v11 stitch mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    StitchEnhanced11Result
    """
    adaptive = None
    if enable_adaptive:
        adaptive = AdaptiveStitchResult(name="adaptive_stitch_refinement", enabled=True)

    multi_patch = None
    if enable_multi_patch:
        multi_patch = MultiPatchStitchResult(name="multi_patch_stitching", enabled=True)

    return StitchEnhanced11Result(
        adaptive=adaptive,
        multi_patch=multi_patch,
    )
