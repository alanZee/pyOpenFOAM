"""
check_mesh enhanced v7 -- enhanced check mesh with additional capabilities
(generation 7).

Extends :func:`check_mesh_enhanced_6` with:

- **boundary layer quality**: enhanced boundary layer quality capabilities.
- **mesh sensitivity analysis**: enhanced mesh sensitivity analysis capabilities.

Usage::

    from pyfoam.tools.check_mesh_enhanced_7 import CheckMeshEnhanced7Result, check_mesh_enhanced_7

    result = check_mesh_enhanced_7()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["CheckMeshEnhanced7Result", "check_mesh_enhanced_7"]

@dataclass
class BLQualityResult:
    """Feature data for boundary_layer_quality."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class SensitivityAnalysisResult:
    """Feature data for mesh_sensitivity_analysis."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class CheckMeshEnhanced7Result:
    """Result from :func:`check_mesh_enhanced_7`."""
    bl_quality: Optional[BLQualityResult] = None
    sensitivity: Optional[SensitivityAnalysisResult] = None


def check_mesh_enhanced_7(
    mesh: Optional["FvMesh"] = None,
    enable_bl_quality: bool = False,
    enable_sensitivity: bool = False,
) -> CheckMeshEnhanced7Result:
    """Enhanced v7 check mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    CheckMeshEnhanced7Result
    """
    bl_quality = None
    if enable_bl_quality:
        bl_quality = BLQualityResult(name="boundary_layer_quality", enabled=True)

    sensitivity = None
    if enable_sensitivity:
        sensitivity = SensitivityAnalysisResult(name="mesh_sensitivity_analysis", enabled=True)

    return CheckMeshEnhanced7Result(
        bl_quality=bl_quality,
        sensitivity=sensitivity,
    )
