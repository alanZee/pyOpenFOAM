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
LimitedLinearInterpolation
    TVD limited-linear with flux limiter (vanLeer/minmod/superbee).
FilteredLinearInterpolation
    NVD-filtered linear: linear clipped to cell min/max range.
BlendedInterpolation
    Convex combination of two interpolation schemes.
LinearFit2Interpolation
    Second variant of linear fit with distance-squared weighting.
CubicUpwindInterpolation
    Cubic upwind-biased scheme for structured meshes.
AMIInterpolation
    Arbitrary Mesh Interface interpolation for non-conformal interfaces.
LinearUpwindFitInterpolation
    Linear upwind with least-squares fit gradient reconstruction.
UpwindFitInterpolation
    Upwind with least-squares fit gradient reconstruction.
CubicUpwindFitInterpolation
    Cubic upwind with least-squares fit for structured meshes.
FilteredLinear2Interpolation
    Second variant of filtered linear with tighter NVD bounding.
FilteredLinearVInterpolation
    Vector variant of filtered linear (component-wise NVD clipping).
VanLeerVInterpolation
    Vector variant of Van Leer TVD scheme.
MUSCLVInterpolation
    Vector variant of MUSCL TVD scheme with minmod limiter.
GammaVInterpolation
    Vector variant of Gamma Peclet-number-based blending.
ClippedLinearInterpolation
    Linear clipped to cell min/max (equivalent to SFCD).
CorrectedLinearInterpolation
    Linear with gradient-based non-orthogonal correction.
LinearUpwindFit2Interpolation
    v2 linear upwind with distance-squared weighted least-squares fit.
UpwindFit2Interpolation
    v2 upwind with averaged gradient reconstruction.
CubicUpwindFit2Interpolation
    v2 cubic upwind with distance-squared blending.
FilteredLinear3Interpolation
    Third variant with Peclet-dependent adaptive NVD filtering.
VanLeerV2Interpolation
    v2 vector variant blending Van Leer with superbee limiter.
MUSCLV2Interpolation
    v2 vector variant using van Albada limiter.
GammaV2Interpolation
    v2 vector variant with adjustable Peclet exponent blending.

Time derivative (ddt) schemes
-----------------------------
EulerDdt
    First-order implicit Euler (backward Euler).
SteadyStateDdt
    Zero time derivative for steady-state solvers.
CrankNicolsonDdt
    Second-order Crank-Nicolson with blending coefficient.
BackwardDdt
    Second-order backward differencing (BDF2) with three time levels.
BackwardDdt2
    v2 BDF2 with variable-dt support (generalised BDF2).
BoundedDdt
    Bounded Euler with Courant-number-based limiting.
BoundedDdt2
    v2 bounded Euler with smooth adaptive limiting.

Gradient schemes
----------------
GaussLinearGrad
    Gauss theorem with linear face interpolation (default).
LeastSquaresGrad
    Least-squares gradient reconstruction.
FourthGrad
    Fourth-order gradient using extended stencil.
FourthGrad2
    v2 fourth-order gradient with enhanced tangential correction.
CellLimitedGrad
    Cell-limited gradient to prevent overshoots.
CellLimitedGrad2
    v2 cell-limited gradient with face-smoothed limiter.
FaceLimitedGrad
    Face-limited gradient.
FaceLimitedGrad2
    v2 face-limited gradient with higher blending ratio.
GaussLinearCorrectedGrad
    Gauss linear with non-orthogonal correction.

Surface-normal gradient schemes
-------------------------------
UncorrectedSnGrad
    Simple (φ_N − φ_P)·δ — exact for orthogonal meshes.
CorrectedSnGrad
    Full non-orthogonal correction using cell gradient.
LimitedSnGrad
    Limited non-orthogonal correction with coefficient.
OrthogonalSnGrad
    Simple orthogonal snGrad — fast path for orthogonal meshes.
OrthogonalSnGrad2
    v2 orthogonal snGrad with inverse distance weighting.
OverRelaxedSnGrad
    Over-relaxed correction for non-orthogonal meshes.
OverRelaxedSnGrad2
    v2 over-relaxed snGrad with squared correction.
BoundedSnGrad
    Bounded snGrad to prevent overshoots.
BoundedSnGrad2
    v2 bounded snGrad with wider bounding range.

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
from pyfoam.discretisation.schemes.linear_upwind_fit_3 import LinearUpwindFit3Interpolation
from pyfoam.discretisation.schemes.upwind_fit_3 import UpwindFit3Interpolation
from pyfoam.discretisation.schemes.cubic_upwind_fit_3 import CubicUpwindFit3Interpolation
from pyfoam.discretisation.schemes.filtered_linear_4 import FilteredLinear4Interpolation
from pyfoam.discretisation.schemes.van_leer_v_3 import VanLeerV3Interpolation
from pyfoam.discretisation.schemes.muscl_v_3 import MUSCLV3Interpolation
from pyfoam.discretisation.schemes.gamma_v_3 import GammaV3Interpolation
from pyfoam.discretisation.schemes.linear_upwind_fit_4 import LinearUpwindFit4Interpolation
from pyfoam.discretisation.schemes.upwind_fit_4 import UpwindFit4Interpolation
from pyfoam.discretisation.schemes.cubic_upwind_fit_4 import CubicUpwindFit4Interpolation
from pyfoam.discretisation.schemes.filtered_linear_5 import FilteredLinear5Interpolation
from pyfoam.discretisation.schemes.van_leer_v_4 import VanLeerV4Interpolation
from pyfoam.discretisation.schemes.muscl_v_4 import MUSCLV4Interpolation
from pyfoam.discretisation.schemes.gamma_v_4 import GammaV4Interpolation
from pyfoam.discretisation.ddt import (
    DdtScheme,
    EulerDdt,
    SteadyStateDdt,
    CrankNicolsonDdt,
    BackwardDdt,
    BackwardDdt2,
    BackwardDdt3,
    BackwardDdt4,
    BoundedDdt,
    BoundedDdt2,
    BoundedDdt3,
    BoundedDdt4,
    DDT_REGISTRY,
    create_ddt_scheme,
)
from pyfoam.discretisation.grad import (
    GradScheme,
    GaussLinearGrad,
    LeastSquaresGrad,
    FourthGrad,
    FourthGrad2,
    FourthGrad3,
    FourthGrad4,
    CellLimitedGrad,
    CellLimitedGrad2,
    CellLimitedGrad3,
    CellLimitedGrad4,
    FaceLimitedGrad,
    FaceLimitedGrad2,
    FaceLimitedGrad3,
    FaceLimitedGrad4,
    GaussLinearCorrectedGrad,
    resolve_grad_scheme,
)
from pyfoam.discretisation.sn_grad import (
    SnGradScheme,
    UncorrectedSnGrad,
    CorrectedSnGrad,
    LimitedSnGrad,
    OrthogonalSnGrad,
    OrthogonalSnGrad2,
    OrthogonalSnGrad3,
    OverRelaxedSnGrad,
    OverRelaxedSnGrad2,
    OverRelaxedSnGrad3,
    BoundedSnGrad,
    BoundedSnGrad2,
    BoundedSnGrad3,
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
    "LinearUpwindFit3Interpolation",
    "UpwindFit3Interpolation",
    "CubicUpwindFit3Interpolation",
    "FilteredLinear4Interpolation",
    "VanLeerV3Interpolation",
    "MUSCLV3Interpolation",
    "GammaV3Interpolation",
    # Time derivative (ddt) schemes
    "DdtScheme",
    "EulerDdt",
    "SteadyStateDdt",
    "CrankNicolsonDdt",
    "BackwardDdt",
    "BackwardDdt2",
    "BackwardDdt3",
    "BoundedDdt",
    "BoundedDdt2",
    "BoundedDdt3",
    "DDT_REGISTRY",
    "create_ddt_scheme",
    # Gradient schemes
    "GradScheme",
    "GaussLinearGrad",
    "LeastSquaresGrad",
    "FourthGrad",
    "FourthGrad2",
    "FourthGrad3",
    "CellLimitedGrad",
    "CellLimitedGrad2",
    "CellLimitedGrad3",
    "FaceLimitedGrad",
    "FaceLimitedGrad2",
    "FaceLimitedGrad3",
    "GaussLinearCorrectedGrad",
    "resolve_grad_scheme",
    # Surface-normal gradient schemes
    "SnGradScheme",
    "UncorrectedSnGrad",
    "CorrectedSnGrad",
    "LimitedSnGrad",
    "OrthogonalSnGrad",
    "OrthogonalSnGrad2",
    "OrthogonalSnGrad3",
    "OverRelaxedSnGrad",
    "OverRelaxedSnGrad2",
    "OverRelaxedSnGrad3",
    "BoundedSnGrad",
    "BoundedSnGrad2",
    "BoundedSnGrad3",
    "sn_grad_from_name",
]
