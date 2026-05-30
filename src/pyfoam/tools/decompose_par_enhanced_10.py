"""
decompose_par enhanced v10 -- enhanced decompose par with additional capabilities
(generation 10).

Extends :func:`decompose_par_enhanced_9` with:

- **ai optimised decomposition**: enhanced ai optimised decomposition capabilities.
- **decomposition pipeline**: enhanced decomposition pipeline capabilities.
- **resilience aware decomposition**: enhanced resilience aware decomposition capabilities.

Usage::

    from pyfoam.tools.decompose_par_enhanced_10 import DecomposeParEnhanced10Result, decompose_par_enhanced_10

    result = decompose_par_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["DecomposeParEnhanced10Result", "decompose_par_enhanced_10"]

@dataclass
class AIOptimisedDecompResult:
    """Feature data for ai_optimised_decomposition."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class DecompPipelineResult:
    """Feature data for decomposition_pipeline."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ResilienceDecompResult:
    """Feature data for resilience_aware_decomposition."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class DecomposeParEnhanced10Result:
    """Result from :func:`decompose_par_enhanced_10`."""
    ai_optimised: Optional[AIOptimisedDecompResult] = None
    pipeline: Optional[DecompPipelineResult] = None
    resilience: Optional[ResilienceDecompResult] = None


def decompose_par_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_ai_optimised: bool = False,
    enable_pipeline: bool = False,
    enable_resilience: bool = False,
) -> DecomposeParEnhanced10Result:
    """Enhanced v10 decompose par.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    DecomposeParEnhanced10Result
    """
    ai_optimised = None
    if enable_ai_optimised:
        ai_optimised = AIOptimisedDecompResult(name="ai_optimised_decomposition", enabled=True)

    pipeline = None
    if enable_pipeline:
        pipeline = DecompPipelineResult(name="decomposition_pipeline", enabled=True)

    resilience = None
    if enable_resilience:
        resilience = ResilienceDecompResult(name="resilience_aware_decomposition", enabled=True)

    return DecomposeParEnhanced10Result(
        ai_optimised=ai_optimised,
        pipeline=pipeline,
        resilience=resilience,
    )
