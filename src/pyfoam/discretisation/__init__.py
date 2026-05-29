"""
pyfoam.discretisation — Finite volume discretisation schemes.

Provides face interpolation schemes for converting cell-centre values
to face values, required for flux calculations in divergence operators.

Schemes
-------
LinearInterpolation
    Second-order linear interpolation: φ_f = w·φ_P + (1-w)·φ_N.
UpwindInterpolation
    First-order upwind based on face flux direction.
LinearUpwindInterpolation
    Second-order upwind-biased interpolation.
QuickInterpolation
    Third-order QUICK scheme with deferred correction.
HarmonicInterpolation
    Harmonic mean interpolation for diffusivity fields.
MidPointInterpolation
    Unweighted arithmetic average (weight = 0.5).
LUSTInterpolation
    LUST blend: 0.75 * linear + 0.25 * linearUpwind.
VanLeerInterpolation
    TVD scheme with van Leer flux limiter.
GammaInterpolation
    Peclet-number-based blending of upwind and linear.
InterfaceCompressionInterpolation
    VOF compressive scheme for interface sharpening.
MUSCLInterpolation
    TVD scheme with minmod limiter for monotonicity.
CentralInterpolation
    Central difference (explicit alias for linear interpolation).
SFCDInterpolation
    Self-Filtered Central Difference: linear clipped to cell min/max.
CubicInterpolation
    Fourth-order cubic with gradient-based correction.
LinearFitInterpolation
    Weighted least-squares linear fit reconstruction.

Time derivative (ddt) schemes
-----------------------------
EulerDdt
    First-order implicit Euler (backward Euler).
SteadyStateDdt
    Zero time derivative for steady-state solvers.
CrankNicolsonDdt
    Second-order Crank-Nicolson with blending coefficient.

Gradient schemes
----------------
GaussLinearGrad
    Gauss theorem with linear face interpolation (default).
LeastSquaresGrad
    Least-squares gradient reconstruction.

Surface-normal gradient schemes
-------------------------------
UncorrectedSnGrad
    Simple (φ_N − φ_P)·δ — exact for orthogonal meshes.
CorrectedSnGrad
    Full non-orthogonal correction using cell gradient.
LimitedSnGrad
    Limited non-orthogonal correction with coefficient.

Weight utilities
----------------
compute_centre_weights
    Distance-based linear interpolation weights.
compute_upwind_weights
    Binary upwind weights from face flux direction.
"""

from pyfoam.discretisation.weights import (
    compute_centre_weights,
    compute_upwind_weights,
)
from pyfoam.discretisation.interpolation import (
    InterpolationScheme,
    LinearInterpolation,
)
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
from pyfoam.discretisation.ddt import (
    DdtScheme,
    EulerDdt,
    SteadyStateDdt,
    CrankNicolsonDdt,
    DDT_REGISTRY,
    create_ddt_scheme,
)
from pyfoam.discretisation.grad import (
    GradScheme,
    GaussLinearGrad,
    LeastSquaresGrad,
    resolve_grad_scheme,
)
from pyfoam.discretisation.sn_grad import (
    SnGradScheme,
    UncorrectedSnGrad,
    CorrectedSnGrad,
    LimitedSnGrad,
    sn_grad_from_name,
)

__all__ = [
    # Weights
    "compute_centre_weights",
    "compute_upwind_weights",
    # Interpolation base
    "InterpolationScheme",
    # Interpolation schemes
    "LinearInterpolation",
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
    # Time derivative (ddt) schemes
    "DdtScheme",
    "EulerDdt",
    "SteadyStateDdt",
    "CrankNicolsonDdt",
    "DDT_REGISTRY",
    "create_ddt_scheme",
    # Gradient schemes
    "GradScheme",
    "GaussLinearGrad",
    "LeastSquaresGrad",
    "resolve_grad_scheme",
    # Surface-normal gradient schemes
    "SnGradScheme",
    "UncorrectedSnGrad",
    "CorrectedSnGrad",
    "LimitedSnGrad",
    "sn_grad_from_name",
]
