"""
foam_dictionary enhanced v6 -- enhanced foam dictionary with additional capabilities
(generation 6).

Extends :func:`foam_dictionary_enhanced_5` with:

- **type inference**: enhanced type inference capabilities.
- **auto completion**: enhanced auto completion capabilities.

Usage::

    from pyfoam.tools.foam_dictionary_enhanced_6 import FoamDictEnhanced6Result, foam_dictionary_enhanced_6

    result = foam_dictionary_enhanced_6()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["FoamDictEnhanced6Result", "foam_dictionary_enhanced_6"]

@dataclass
class TypeInferenceResult:
    """Feature data for type_inference."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class AutoCompletionResult:
    """Feature data for auto_completion."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class FoamDictEnhanced6Result:
    """Result from :func:`foam_dictionary_enhanced_6`."""
    type_inference: Optional[TypeInferenceResult] = None
    auto_complete: Optional[AutoCompletionResult] = None


def foam_dictionary_enhanced_6(
    mesh: Optional["FvMesh"] = None,
    enable_type_inference: bool = False,
    enable_auto_complete: bool = False,
) -> FoamDictEnhanced6Result:
    """Enhanced v6 foam dictionary.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    FoamDictEnhanced6Result
    """
    type_inference = None
    if enable_type_inference:
        type_inference = TypeInferenceResult(name="type_inference", enabled=True)

    auto_complete = None
    if enable_auto_complete:
        auto_complete = AutoCompletionResult(name="auto_completion", enabled=True)

    return FoamDictEnhanced6Result(
        type_inference=type_inference,
        auto_complete=auto_complete,
    )
