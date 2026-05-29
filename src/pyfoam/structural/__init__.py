"""
pyfoam.structural — Structural mechanics solvers.

Provides:

- :class:`LinearElasticModel` — isotropic linear elastic constitutive model
- :class:`VonMisesYield` — von Mises yield criterion
- :class:`StressSolver` — stress field computation
- :class:`DisplacementSolver` — displacement field computation
- :class:`AnisotropicElasticModel` — fully anisotropic elastic model
- :class:`OrthotropicElasticModel` — orthotropic elastic model
- :class:`IsotropicPlasticModel` — isotropic elastic + hardening plasticity
- :class:`EnhancedStressSolver` — iterative stress solver with nonlinear support
- :class:`EnhancedDisplacementSolver` — displacement solver with large deformation
"""

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver import StressSolver
from pyfoam.structural.displacement_solver import DisplacementSolver
from pyfoam.structural.elastic_model_enhanced import (
    AnisotropicElasticModel,
    OrthotropicElasticModel,
    IsotropicPlasticModel,
)
from pyfoam.structural.stress_solver_enhanced import (
    EnhancedStressSolver,
    IterativeStressResult,
)
from pyfoam.structural.displacement_solver_enhanced import (
    EnhancedDisplacementSolver,
    NonlinearSolveResult,
)
from pyfoam.structural.elastic_model_enhanced_2 import (
    TransverselyIsotropicModel,
    HyperelasticNeoHookean,
    CombinedPlasticityModel,
)
from pyfoam.structural.stress_solver_enhanced_2 import (
    EnhancedStressSolver2,
    AdaptiveStressResult,
)
from pyfoam.structural.displacement_solver_enhanced_2 import (
    EnhancedDisplacementSolver2,
    ArcLengthResult,
    LoadStepResult,
)

__all__ = [
    "LinearElasticModel",
    "VonMisesYield",
    "StressSolver",
    "DisplacementSolver",
    # Enhanced
    "AnisotropicElasticModel",
    "OrthotropicElasticModel",
    "IsotropicPlasticModel",
    "EnhancedStressSolver",
    "IterativeStressResult",
    "EnhancedDisplacementSolver",
    "NonlinearSolveResult",
    # V2 enhanced
    "TransverselyIsotropicModel",
    "HyperelasticNeoHookean",
    "CombinedPlasticityModel",
    "EnhancedStressSolver2",
    "AdaptiveStressResult",
    "EnhancedDisplacementSolver2",
    "ArcLengthResult",
    "LoadStepResult",
]
