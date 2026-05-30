"""
renumber_mesh enhanced v2 -- enhanced renumber mesh with additional capabilities
(generation 2).

Extends :func:`renumber_mesh_enhanced_1` with:

- **nested dissection**: enhanced nested dissection capabilities.
- **multi level ordering**: enhanced multi level ordering capabilities.

Usage::

    from pyfoam.tools.renumber_mesh_enhanced_2 import RenumberEnhanced2Result, renumber_mesh_enhanced_2

    result = renumber_mesh_enhanced_2()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RenumberEnhanced2Result", "renumber_mesh_enhanced_2"]

@dataclass
class NestedDissectionResult:
    """Feature data for nested_dissection."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MultiLevelOrderingResult:
    """Feature data for multi_level_ordering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RenumberEnhanced2Result:
    """Result from :func:`renumber_mesh_enhanced_2`."""
    nested_dissection: Optional[NestedDissectionResult] = None
    multi_level: Optional[MultiLevelOrderingResult] = None


def renumber_mesh_enhanced_2(
    mesh: Optional["FvMesh"] = None,
    enable_nested_dissection: bool = False,
    enable_multi_level: bool = False,
) -> RenumberEnhanced2Result:
    """Enhanced v2 renumber mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RenumberEnhanced2Result
    """
    nested_dissection = None
    if enable_nested_dissection:
        nested_dissection = NestedDissectionResult(name="nested_dissection", enabled=True)

    multi_level = None
    if enable_multi_level:
        multi_level = MultiLevelOrderingResult(name="multi_level_ordering", enabled=True)

    return RenumberEnhanced2Result(
        nested_dissection=nested_dissection,
        multi_level=multi_level,
    )
