"""
pyfoam.discretisation.schemes — Concrete interpolation schemes.

Contains interpolation schemes that operate as GPU tensor gather/scatter
operations: upwind, linear-upwind, QUICK, harmonic, midPoint, LUST,
vanLeer, gamma, interfaceCompression, MUSCL, central, SFCD, cubic,
linearFit, limitedLinear, filteredLinear, blended, linearFit2,
cubicUpwind, and AMI.  Also includes v2 variants and filteredLinear3.
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
from pyfoam.discretisation.schemes.linear_upwind_fit import LinearUpwindFitInterpolation
from pyfoam.discretisation.schemes.upwind_fit import UpwindFitInterpolation
from pyfoam.discretisation.schemes.cubic_upwind_fit import CubicUpwindFitInterpolation
from pyfoam.discretisation.schemes.filtered_linear_2 import FilteredLinear2Interpolation
from pyfoam.discretisation.schemes.filtered_linear_v import FilteredLinearVInterpolation
from pyfoam.discretisation.schemes.van_leer_v import VanLeerVInterpolation
from pyfoam.discretisation.schemes.muscl_v import MUSCLVInterpolation
from pyfoam.discretisation.schemes.gamma_v import GammaVInterpolation
from pyfoam.discretisation.schemes.clipped_linear import ClippedLinearInterpolation
from pyfoam.discretisation.schemes.corrected_linear import CorrectedLinearInterpolation
from pyfoam.discretisation.schemes.linear_upwind_fit_2 import LinearUpwindFit2Interpolation
from pyfoam.discretisation.schemes.upwind_fit_2 import UpwindFit2Interpolation
from pyfoam.discretisation.schemes.cubic_upwind_fit_2 import CubicUpwindFit2Interpolation
from pyfoam.discretisation.schemes.filtered_linear_3 import FilteredLinear3Interpolation
from pyfoam.discretisation.schemes.van_leer_v_2 import VanLeerV2Interpolation
from pyfoam.discretisation.schemes.muscl_v_2 import MUSCLV2Interpolation
from pyfoam.discretisation.schemes.gamma_v_2 import GammaV2Interpolation

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
    "LinearUpwindFitInterpolation",
    "UpwindFitInterpolation",
    "CubicUpwindFitInterpolation",
    "FilteredLinear2Interpolation",
    "FilteredLinearVInterpolation",
    "VanLeerVInterpolation",
    "MUSCLVInterpolation",
    "GammaVInterpolation",
    "ClippedLinearInterpolation",
    "CorrectedLinearInterpolation",
    "LinearUpwindFit2Interpolation",
    "UpwindFit2Interpolation",
    "CubicUpwindFit2Interpolation",
    "FilteredLinear3Interpolation",
    "VanLeerV2Interpolation",
    "MUSCLV2Interpolation",
    "GammaV2Interpolation",
]
