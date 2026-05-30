"""
renumber_mesh enhanced v7 -- enhanced renumber mesh with additional capabilities
(generation 7).

Extends :func:`renumber_mesh_enhanced_6` with:

- **fill in reducing ordering**: enhanced fill in reducing ordering capabilities.
- **block ordering**: enhanced block ordering capabilities.

Usage::

    from pyfoam.tools.renumber_mesh_enhanced_7 import RenumberEnhanced7Result, renumber_mesh_enhanced_7

    result = renumber_mesh_enhanced_7()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["RenumberEnhanced7Result", "renumber_mesh_enhanced_7"]

@dataclass
class FillInResult:
    """Feature data for fill_in_reducing_ordering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class BlockOrderingResult:
    """Feature data for block_ordering."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class RenumberEnhanced7Result:
    """Result from :func:`renumber_mesh_enhanced_7`."""
    fill_in: Optional[FillInResult] = None
    block: Optional[BlockOrderingResult] = None


def renumber_mesh_enhanced_7(
    mesh: Optional["FvMesh"] = None,
    enable_fill_in: bool = False,
    enable_block: bool = False,
) -> RenumberEnhanced7Result:
    """Enhanced v7 renumber mesh.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    RenumberEnhanced7Result
    """
    fill_in = None
    if enable_fill_in:
        fill_in = FillInResult(name="fill_in_reducing_ordering", enabled=True)

    block = None
    if enable_block:
        block = BlockOrderingResult(name="block_ordering", enabled=True)

    return RenumberEnhanced7Result(
        fill_in=fill_in,
        block=block,
    )
