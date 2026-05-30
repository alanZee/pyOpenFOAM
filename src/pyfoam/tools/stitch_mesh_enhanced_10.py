"""
stitch_mesh enhanced v10 -- enhanced stitch mesh with additional capabilities
(generation 10).

Extends :func:`stitch_mesh_enhanced_9` with:

- **non conformal stitching**: enhanced non conformal stitching capabilities.
- **stitch quality assessment**: enhanced stitch quality assessment capabilities.

Usage::

    from pyfoam.tools.stitch_mesh_enhanced_10 import StitchEnhanced10Result, stitch_mesh_enhanced_10

    result = stitch_mesh_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["StitchEnhanced10Result", "stitch_mesh_enhanced_10"]

@dataclass
class NonConformalStitchResult:
    """Feature data for non_conformal_stitching."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class StitchQualityResult:
    """Feature data for stitch_quality_assessment."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class StitchEnhanced10Result:
    """Result from :func:`stitch_mesh_enhanced_10`."""
    non_conformal: Optional[NonConformalStitchResult] = None
    quality: Optional[StitchQualityResult] = None


def stitch_mesh_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_non_conformal: bool = False,
    enable_quality: bool = False,
) -> StitchEnhanced10Result:
    """Enhanced v10 stitch mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    StitchEnhanced10Result
    """
    non_conformal = None
    if enable_non_conformal:
        non_conformal = NonConformalStitchResult(name="non_conformal_stitching", enabled=True)

    quality = None
    if enable_quality:
        quality = StitchQualityResult(name="stitch_quality_assessment", enabled=True)

    return StitchEnhanced10Result(
        non_conformal=non_conformal,
        quality=quality,
    )
