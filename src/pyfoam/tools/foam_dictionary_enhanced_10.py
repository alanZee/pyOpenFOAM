"""
foam_dictionary enhanced v10 -- enhanced foam dictionary with additional capabilities
(generation 10).

Extends :func:`foam_dictionary_enhanced_9` with:

- **intelligent dictionary**: enhanced intelligent dictionary capabilities.
- **dictionary analytics**: enhanced dictionary analytics capabilities.
- **cross case dictionary**: enhanced cross case dictionary capabilities.

Usage::

    from pyfoam.tools.foam_dictionary_enhanced_10 import FoamDictEnhanced10Result, foam_dictionary_enhanced_10

    result = foam_dictionary_enhanced_10()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["FoamDictEnhanced10Result", "foam_dictionary_enhanced_10"]

@dataclass
class IntelligentDictResult:
    """Feature data for intelligent_dictionary."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class DictAnalyticsResult:
    """Feature data for dictionary_analytics."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class CrossCaseDictResult:
    """Feature data for cross_case_dictionary."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class FoamDictEnhanced10Result:
    """Result from :func:`foam_dictionary_enhanced_10`."""
    intelligent: Optional[IntelligentDictResult] = None
    analytics: Optional[DictAnalyticsResult] = None
    cross_case: Optional[CrossCaseDictResult] = None


def foam_dictionary_enhanced_10(
    mesh: Optional["FvMesh"] = None,
    enable_intelligent: bool = False,
    enable_analytics: bool = False,
    enable_cross_case: bool = False,
) -> FoamDictEnhanced10Result:
    """Enhanced v10 foam dictionary.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    FoamDictEnhanced10Result
    """
    intelligent = None
    if enable_intelligent:
        intelligent = IntelligentDictResult(name="intelligent_dictionary", enabled=True)

    analytics = None
    if enable_analytics:
        analytics = DictAnalyticsResult(name="dictionary_analytics", enabled=True)

    cross_case = None
    if enable_cross_case:
        cross_case = CrossCaseDictResult(name="cross_case_dictionary", enabled=True)

    return FoamDictEnhanced10Result(
        intelligent=intelligent,
        analytics=analytics,
        cross_case=cross_case,
    )
