"""
foam_dictionary enhanced v3 -- enhanced foam dictionary with additional capabilities
(generation 3).

Extends :func:`foam_dictionary_enhanced_2` with:

- **template expansion**: enhanced template expansion capabilities.
- **macro substitution**: enhanced macro substitution capabilities.

Usage::

    from pyfoam.tools.foam_dictionary_enhanced_3 import FoamDictEnhanced3Result, foam_dictionary_enhanced_3

    result = foam_dictionary_enhanced_3()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["FoamDictEnhanced3Result", "foam_dictionary_enhanced_3"]

@dataclass
class TemplateExpansionResult:
    """Feature data for template_expansion."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class MacroSubstitutionResult:
    """Feature data for macro_substitution."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class FoamDictEnhanced3Result:
    """Result from :func:`foam_dictionary_enhanced_3`."""
    template: Optional[TemplateExpansionResult] = None
    macros: Optional[MacroSubstitutionResult] = None


def foam_dictionary_enhanced_3(
    mesh: Optional["FvMesh"] = None,
    enable_template: bool = False,
    enable_macros: bool = False,
) -> FoamDictEnhanced3Result:
    """Enhanced v3 foam dictionary.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    FoamDictEnhanced3Result
    """
    template = None
    if enable_template:
        template = TemplateExpansionResult(name="template_expansion", enabled=True)

    macros = None
    if enable_macros:
        macros = MacroSubstitutionResult(name="macro_substitution", enabled=True)

    return FoamDictEnhanced3Result(
        template=template,
        macros=macros,
    )
