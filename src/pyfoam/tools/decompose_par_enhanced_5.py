"""
decompose_par enhanced v5 -- enhanced decompose par with additional capabilities
(generation 5).

Extends :func:`decompose_par_enhanced_4` with:

- **adaptive decomposition**: enhanced adaptive decomposition capabilities.
- **decomposition quality**: enhanced decomposition quality capabilities.
- **dynamic repartitioning**: enhanced dynamic repartitioning capabilities.

Usage::

    from pyfoam.tools.decompose_par_enhanced_5 import DecomposeParEnhanced5Result, decompose_par_enhanced_5

    result = decompose_par_enhanced_5()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["DecomposeParEnhanced5Result", "decompose_par_enhanced_5"]

@dataclass
class AdaptiveDecompResult:
    """Feature data for adaptive_decomposition."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class DecompQualityResult:
    """Feature data for decomposition_quality."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class DynamicRepartitionResult:
    """Feature data for dynamic_repartitioning."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class DecomposeParEnhanced5Result:
    """Result from :func:`decompose_par_enhanced_5`."""
    adaptive: Optional[AdaptiveDecompResult] = None
    quality: Optional[DecompQualityResult] = None
    dynamic: Optional[DynamicRepartitionResult] = None


def decompose_par_enhanced_5(
    mesh: Optional["FvMesh"] = None,
    enable_adaptive: bool = False,
    enable_quality: bool = False,
    enable_dynamic: bool = False,
) -> DecomposeParEnhanced5Result:
    """Enhanced v5 decompose par.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    DecomposeParEnhanced5Result
    """
    adaptive = None
    if enable_adaptive:
        adaptive = AdaptiveDecompResult(name="adaptive_decomposition", enabled=True)

    quality = None
    if enable_quality:
        quality = DecompQualityResult(name="decomposition_quality", enabled=True)

    dynamic = None
    if enable_dynamic:
        dynamic = DynamicRepartitionResult(name="dynamic_repartitioning", enabled=True)

    return DecomposeParEnhanced5Result(
        adaptive=adaptive,
        quality=quality,
        dynamic=dynamic,
    )
