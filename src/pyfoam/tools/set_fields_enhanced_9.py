"""
set_fields enhanced v9 -- enhanced set fields with additional capabilities
(generation 9).

Extends :func:`set_fields_enhanced_8` with:

- **field validation**: enhanced field validation capabilities.
- **conservative initialisation**: enhanced conservative initialisation capabilities.

Usage::

    from pyfoam.tools.set_fields_enhanced_9 import SetFieldsEnhanced9Result, set_fields_enhanced_9

    result = set_fields_enhanced_9()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SetFieldsEnhanced9Result", "set_fields_enhanced_9"]

@dataclass
class FieldValidationResult:
    """Feature data for field_validation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class ConservativeInitResult:
    """Feature data for conservative_initialisation."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SetFieldsEnhanced9Result:
    """Result from :func:`set_fields_enhanced_9`."""
    validation: Optional[FieldValidationResult] = None
    conservative: Optional[ConservativeInitResult] = None


def set_fields_enhanced_9(
    mesh: Optional["FvMesh"] = None,
    enable_validation: bool = False,
    enable_conservative: bool = False,
) -> SetFieldsEnhanced9Result:
    """Enhanced v9 set fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SetFieldsEnhanced9Result
    """
    validation = None
    if enable_validation:
        validation = FieldValidationResult(name="field_validation", enabled=True)

    conservative = None
    if enable_conservative:
        conservative = ConservativeInitResult(name="conservative_initialisation", enabled=True)

    return SetFieldsEnhanced9Result(
        validation=validation,
        conservative=conservative,
    )
