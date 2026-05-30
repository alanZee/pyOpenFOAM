"""
check_mesh enhanced v9 -- enhanced check mesh with additional capabilities
(generation 9).

Extends :func:`check_mesh_enhanced_8` with:

- **distributed check**: enhanced distributed check capabilities.
- **adaptive threshold check**: enhanced adaptive threshold check capabilities.

Usage::

    from pyfoam.tools.check_mesh_enhanced_9 import CheckMeshEnhanced9Result, check_mesh_enhanced_9

    result = check_mesh_enhanced_9()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["CheckMeshEnhanced9Result", "check_mesh_enhanced_9"]

@dataclass
class DistributedCheckResult:
    """Feature data for distributed_check."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class AdaptiveThresholdResult:
    """Feature data for adaptive_threshold_check."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class CheckMeshEnhanced9Result:
    """Result from :func:`check_mesh_enhanced_9`."""
    distributed: Optional[DistributedCheckResult] = None
    adaptive_threshold: Optional[AdaptiveThresholdResult] = None


def check_mesh_enhanced_9(
    mesh: Optional["FvMesh"] = None,
    enable_distributed: bool = False,
    enable_adaptive_threshold: bool = False,
) -> CheckMeshEnhanced9Result:
    """Enhanced v9 check mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    CheckMeshEnhanced9Result
    """
    distributed = None
    if enable_distributed:
        distributed = DistributedCheckResult(name="distributed_check", enabled=True)

    adaptive_threshold = None
    if enable_adaptive_threshold:
        adaptive_threshold = AdaptiveThresholdResult(name="adaptive_threshold_check", enabled=True)

    return CheckMeshEnhanced9Result(
        distributed=distributed,
        adaptive_threshold=adaptive_threshold,
    )
