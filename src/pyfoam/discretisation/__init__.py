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

__all__ = [
    # Weights
    "compute_centre_weights",
    "compute_upwind_weights",
    # Base
    "InterpolationScheme",
    # Schemes
    "LinearInterpolation",
    "UpwindInterpolation",
    "LinearUpwindInterpolation",
    "QuickInterpolation",
]
