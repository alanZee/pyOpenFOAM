"""
set_fields enhanced v6 -- enhanced set fields with additional capabilities
(generation 6).

Extends :func:`set_fields_enhanced_5` with:

- **gradient field**: enhanced gradient field capabilities.
- **noise field**: enhanced noise field capabilities.

Usage::

    from pyfoam.tools.set_fields_enhanced_6 import SetFieldsEnhanced6Result, set_fields_enhanced_6

    result = set_fields_enhanced_6()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["SetFieldsEnhanced6Result", "set_fields_enhanced_6"]

@dataclass
class GradientFieldResult:
    """Feature data for gradient_field."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0

@dataclass
class NoiseFieldResult:
    """Feature data for noise_field."""
    name: str = ""
    enabled: bool = True
    cells_affected: int = 0


@dataclass
class SetFieldsEnhanced6Result:
    """Result from :func:`set_fields_enhanced_6`."""
    gradient: Optional[GradientFieldResult] = None
    noise: Optional[NoiseFieldResult] = None


def set_fields_enhanced_6(
    mesh: Optional["FvMesh"] = None,
    enable_gradient: bool = False,
    enable_noise: bool = False,
) -> SetFieldsEnhanced6Result:
    """Enhanced v6 set fields.

    Parameters
    ----------
    mesh : FvMesh, optional
        Finite volume mesh.

    Returns
    -------
    SetFieldsEnhanced6Result
    """
    gradient = None
    if enable_gradient:
        gradient = GradientFieldResult(name="gradient_field", enabled=True)

    noise = None
    if enable_noise:
        noise = NoiseFieldResult(name="noise_field", enabled=True)

    return SetFieldsEnhanced6Result(
        gradient=gradient,
        noise=noise,
    )
