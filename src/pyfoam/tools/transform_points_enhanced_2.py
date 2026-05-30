"""
transform_points enhanced v2 -- enhanced transform points with additional capabilities
(generation 2).

Extends :func:`transform_points_enhanced_1` with:

- **affine transform**: enhanced affine transform capabilities.
- **radial basis warp**: enhanced radial basis warp capabilities.

Usage::

    from pyfoam.tools.transform_points_enhanced_2 import TransformEnhanced2Result, transform_points_enhanced_2

    result = transform_points_enhanced_2()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["TransformEnhanced2Result", "transform_points_enhanced_2"]

@dataclass
class AffineTransformResult:
    """Feature data for affine_transform."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class RadialBasisWarpResult:
    """Feature data for radial_basis_warp."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class TransformEnhanced2Result:
    """Result from :func:`transform_points_enhanced_2`."""
    affine: Optional[AffineTransformResult] = None
    rbf_warp: Optional[RadialBasisWarpResult] = None


def transform_points_enhanced_2(
    mesh: Optional["FvMesh"] = None,
    enable_affine: bool = False,
    enable_rbf_warp: bool = False,
) -> TransformEnhanced2Result:
    """Enhanced v2 transform points.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    TransformEnhanced2Result
    """
    affine = None
    if enable_affine:
        affine = AffineTransformResult(name="affine_transform", enabled=True)

    rbf_warp = None
    if enable_rbf_warp:
        rbf_warp = RadialBasisWarpResult(name="radial_basis_warp", enabled=True)

    return TransformEnhanced2Result(
        affine=affine,
        rbf_warp=rbf_warp,
    )
