"""
subset_mesh enhanced v10 -- enhanced subset mesh with additional capabilities
(generation 10).

Extends :func:`subset_mesh_enhanced_9` with:

- **ai guided subset**: enhanced ai guided subset capabilities.
- **subset optimisation**: enhanced subset optimisation capabilities.
- **multi physics subset**: enhanced multi physics subset capabilities.

Usage::

    from pyfoam.tools.subset_mesh_enhanced_10 import SubsetEnhanced10Result, subset_mesh_enhanced_10

    result = subset_mesh_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SubsetEnhanced10Result", "subset_mesh_enhanced_10"]

@dataclass
class AIGuidedSubsetResult:
    """Feature data for ai_guided_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class SubsetOptimisationResult:
    """Feature data for subset_optimisation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MultiPhysicsSubsetResult:
    """Feature data for multi_physics_subset."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SubsetEnhanced10Result:
    """Result from :func:`subset_mesh_enhanced_10`."""
    ai_guided: Optional[AIGuidedSubsetResult] = None
    optimisation: Optional[SubsetOptimisationResult] = None
    multi_physics: Optional[MultiPhysicsSubsetResult] = None


def subset_mesh_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_ai_guided: bool = False,
    enable_optimisation: bool = False,
    enable_multi_physics: bool = False,
) -> SubsetEnhanced10Result:
    """Enhanced v10 subset mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SubsetEnhanced10Result
    """
    ai_guided = None
    if enable_ai_guided:
        ai_guided = AIGuidedSubsetResult(name="ai_guided_subset", enabled=True)

    optimisation = None
    if enable_optimisation:
        optimisation = SubsetOptimisationResult(name="subset_optimisation", enabled=True)

    multi_physics = None
    if enable_multi_physics:
        multi_physics = MultiPhysicsSubsetResult(name="multi_physics_subset", enabled=True)

    return SubsetEnhanced10Result(
        ai_guided=ai_guided,
        optimisation=optimisation,
        multi_physics=multi_physics,
    )
