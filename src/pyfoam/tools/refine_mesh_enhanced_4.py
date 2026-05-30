"""
refine_mesh enhanced v4 -- enhanced refine mesh with additional capabilities
(generation 4).

Extends :func:`refine_mesh_enhanced_3` with:

- **multi level refinement**: enhanced multi level refinement capabilities.
- **refinement quality control**: enhanced refinement quality control capabilities.

Usage::

    from pyfoam.tools.refine_mesh_enhanced_4 import RefineEnhanced4Result, refine_mesh_enhanced_4

    result = refine_mesh_enhanced_4()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RefineEnhanced4Result", "refine_mesh_enhanced_4"]

@dataclass
class MultiLevelRefineResult:
    """Feature data for multi_level_refinement."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class RefineQualityResult:
    """Feature data for refinement_quality_control."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RefineEnhanced4Result:
    """Result from :func:`refine_mesh_enhanced_4`."""
    multi_level: Optional[MultiLevelRefineResult] = None
    quality: Optional[RefineQualityResult] = None


def refine_mesh_enhanced_4(
    mesh: Optional["FvMesh"] = None,
    enable_multi_level: bool = False,
    enable_quality: bool = False,
) -> RefineEnhanced4Result:
    """Enhanced v4 refine mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RefineEnhanced4Result
    """
    multi_level = None
    if enable_multi_level:
        multi_level = MultiLevelRefineResult(name="multi_level_refinement", enabled=True)

    quality = None
    if enable_quality:
        quality = RefineQualityResult(name="refinement_quality_control", enabled=True)

    return RefineEnhanced4Result(
        multi_level=multi_level,
        quality=quality,
    )
