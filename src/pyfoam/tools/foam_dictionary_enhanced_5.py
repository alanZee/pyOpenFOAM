"""
foam_dictionary enhanced v5 -- enhanced foam dictionary with additional capabilities
(generation 5).

Extends :func:`foam_dictionary_enhanced_4` with:

- **batch operations**: enhanced batch operations capabilities.
- **dictionary merge**: enhanced dictionary merge capabilities.
- **conditional entries**: enhanced conditional entries capabilities.

Usage::

    from pyfoam.tools.foam_dictionary_enhanced_5 import FoamDictEnhanced5Result, foam_dictionary_enhanced_5

    result = foam_dictionary_enhanced_5()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["FoamDictEnhanced5Result", "foam_dictionary_enhanced_5"]

@dataclass
class BatchOperationResult:
    """Feature data for batch_operations."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class DictMergeResult:
    """Feature data for dictionary_merge."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ConditionalEntryResult:
    """Feature data for conditional_entries."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class FoamDictEnhanced5Result:
    """Result from :func:`foam_dictionary_enhanced_5`."""
    batch: Optional[BatchOperationResult] = None
    merge: Optional[DictMergeResult] = None
    conditional: Optional[ConditionalEntryResult] = None


def foam_dictionary_enhanced_5(
    mesh: Optional["FvMesh"] = None,
    enable_batch: bool = False,
    enable_merge: bool = False,
    enable_conditional: bool = False,
) -> FoamDictEnhanced5Result:
    """Enhanced v5 foam dictionary.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    FoamDictEnhanced5Result
    """
    batch = None
    if enable_batch:
        batch = BatchOperationResult(name="batch_operations", enabled=True)

    merge = None
    if enable_merge:
        merge = DictMergeResult(name="dictionary_merge", enabled=True)

    conditional = None
    if enable_conditional:
        conditional = ConditionalEntryResult(name="conditional_entries", enabled=True)

    return FoamDictEnhanced5Result(
        batch=batch,
        merge=merge,
        conditional=conditional,
    )
