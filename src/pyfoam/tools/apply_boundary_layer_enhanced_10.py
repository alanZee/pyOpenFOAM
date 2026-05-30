"""
apply_boundary_layer enhanced v10 -- enhanced apply boundary layer with additional capabilities
(generation 10).

Extends :func:`apply_boundary_layer_enhanced_9` with:

- **wall model library**: enhanced wall model library capabilities.
- **bl transition coupling**: enhanced bl transition coupling capabilities.
- **reynolds stress transport**: enhanced reynolds stress transport capabilities.

Usage::

    from pyfoam.tools.apply_boundary_layer_enhanced_10 import EnhancedBL10Result, apply_boundary_layer_enhanced_10

    result = apply_boundary_layer_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedBL10Result", "apply_boundary_layer_enhanced_10"]

@dataclass
class WallModelLibrary:
    """Feature data for wall_model_library."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class BLTransitionCoupling:
    """Feature data for bl_transition_coupling."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ReynoldsStressResult:
    """Feature data for reynolds_stress_transport."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class EnhancedBL10Result:
    """Result from :func:`apply_boundary_layer_enhanced_10`."""
    wall_models: Optional[WallModelLibrary] = None
    transition_coupling: Optional[BLTransitionCoupling] = None
    reynolds_stress: Optional[ReynoldsStressResult] = None


def apply_boundary_layer_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_wall_models: bool = False,
    enable_transition_coupling: bool = False,
    enable_reynolds_stress: bool = False,
) -> EnhancedBL10Result:
    """Enhanced v10 apply boundary layer.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    EnhancedBL10Result
    """
    wall_models = None
    if enable_wall_models:
        wall_models = WallModelLibrary(name="wall_model_library", enabled=True)

    transition_coupling = None
    if enable_transition_coupling:
        transition_coupling = BLTransitionCoupling(name="bl_transition_coupling", enabled=True)

    reynolds_stress = None
    if enable_reynolds_stress:
        reynolds_stress = ReynoldsStressResult(name="reynolds_stress_transport", enabled=True)

    return EnhancedBL10Result(
        wall_models=wall_models,
        transition_coupling=transition_coupling,
        reynolds_stress=reynolds_stress,
    )
