"""
pyfoam.discretisation.schemes — Concrete interpolation schemes.

Contains interpolation schemes that operate as GPU tensor gather/scatter
operations: upwind, linear-upwind, QUICK, harmonic, midPoint, LUST,
vanLeer, gamma, interfaceCompression, MUSCL, central, SFCD, cubic,
linearFit, limitedLinear, filteredLinear, blended, linearFit2,
cubicUpwind, and AMI.
"""

from pyfoam.discretisation.schemes.upwind import UpwindInterpolation
from pyfoam.discretisation.schemes.linear_upwind import LinearUpwindInterpolation
from pyfoam.discretisation.schemes.quick import QuickInterpolation
from pyfoam.discretisation.schemes.harmonic import HarmonicInterpolation
from pyfoam.discretisation.schemes.mid_point import MidPointInterpolation
from pyfoam.discretisation.schemes.lust import LUSTInterpolation
from pyfoam.discretisation.schemes.van_leer import VanLeerInterpolation
from pyfoam.discretisation.schemes.gamma import GammaInterpolation
from pyfoam.discretisation.schemes.interface_compression import InterfaceCompressionInterpolation
from pyfoam.discretisation.schemes.muscl import MUSCLInterpolation
from pyfoam.discretisation.schemes.central import CentralInterpolation
from pyfoam.discretisation.schemes.sfcd import SFCDInterpolation
from pyfoam.discretisation.schemes.cubic import CubicInterpolation
from pyfoam.discretisation.schemes.linear_fit import LinearFitInterpolation
from pyfoam.discretisation.schemes.limited_linear import LimitedLinearInterpolation
from pyfoam.discretisation.schemes.filtered_linear import FilteredLinearInterpolation
from pyfoam.discretisation.schemes.blended import BlendedInterpolation
from pyfoam.discretisation.schemes.linear_fit_2 import LinearFit2Interpolation
from pyfoam.discretisation.schemes.cubic_upwind import CubicUpwindInterpolation
from pyfoam.discretisation.schemes.ami_interpolation import AMIInterpolation

__all__ = [
    "UpwindInterpolation",
    "LinearUpwindInterpolation",
    "QuickInterpolation",
    "HarmonicInterpolation",
    "MidPointInterpolation",
    "LUSTInterpolation",
    "VanLeerInterpolation",
    "GammaInterpolation",
    "InterfaceCompressionInterpolation",
    "MUSCLInterpolation",
    "CentralInterpolation",
    "SFCDInterpolation",
    "CubicInterpolation",
    "LinearFitInterpolation",
    "LimitedLinearInterpolation",
    "FilteredLinearInterpolation",
    "BlendedInterpolation",
    "LinearFit2Interpolation",
    "CubicUpwindInterpolation",
    "AMIInterpolation",
]
