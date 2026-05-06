"""
pyfoam.discretisation.schemes — Concrete interpolation schemes.

Contains upwind, linear-upwind, and QUICK interpolation schemes
that operate as GPU tensor gather/scatter operations.
"""

from pyfoam.discretisation.schemes.upwind import UpwindInterpolation
from pyfoam.discretisation.schemes.linear_upwind import LinearUpwindInterpolation
from pyfoam.discretisation.schemes.quick import QuickInterpolation

__all__ = [
    "UpwindInterpolation",
    "LinearUpwindInterpolation",
    "QuickInterpolation",
]
