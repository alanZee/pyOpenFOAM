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
from pyfoam.structural.elastic_model_enhanced_3 import (
    OrthotropicPlasticModel,
    ViscoelasticMaxwellModel,
    DamageModel,
)
from pyfoam.structural.stress_solver_enhanced_3 import (
    EnhancedStressSolver3,
    NonlinearStressResult,
)
from pyfoam.structural.displacement_solver_enhanced_3 import (
    EnhancedDisplacementSolver3,
    LargeDeformationResult,
)
from pyfoam.structural.elastic_model_enhanced_4 import (
    GursonDamageModel,
    CrystalPlasticityModel,
    PhaseFieldFractureModel,
)
from pyfoam.structural.stress_solver_enhanced_4 import (
    EnhancedStressSolver4,
    SmoothedStressResult,
    ThermalCoupling,
)
from pyfoam.structural.displacement_solver_enhanced_4 import (
    EnhancedDisplacementSolver4,
    ContactResult,
    RefinementIndicator,
)
from pyfoam.structural.elastic_model_enhanced_5 import (
    GradientPlasticityModel,
    CoupledDamagePlasticityModel,
    HyperelasticOgdenModel,
)
from pyfoam.structural.stress_solver_enhanced_5 import (
    EnhancedStressSolver5,
    FailureAssessment,
    StressInvariants,
)
from pyfoam.structural.displacement_solver_enhanced_5 import (
    EnhancedDisplacementSolver5,
    ModalResult,
    NewmarkResult,
    RayleighDamping,
)
from pyfoam.structural.elastic_model_enhanced_6 import (
    ChabocheKinematicHardening,
    JohnsonCookModel,
    ConcreteDamagedPlasticityModel,
)
from pyfoam.structural.stress_solver_enhanced_6 import (
    EnhancedStressSolver6,
    CrackResult,
    FatigueResult,
    CreepResult,
)
from pyfoam.structural.displacement_solver_enhanced_6 import (
    EnhancedDisplacementSolver6,
    BucklingResult,
    ContactResult6,
    GeometricNonlinearResult,
)
from pyfoam.structural.elastic_model_enhanced_7 import (
    ThermomechanicalCouplingModel,
    PorousElasticModel,
    FatigueDamageModel,
)
from pyfoam.structural.stress_solver_enhanced_7 import (
    EnhancedStressSolver7,
    XFEMResult,
    ThermalStressResult,
    HomogenisationResult,
)
from pyfoam.structural.displacement_solver_enhanced_7 import (
    EnhancedDisplacementSolver7,
    TopologyResult,
    RefinementResult7,
    SubstructureResult,
)
from pyfoam.structural.elastic_model_enhanced_8 import (
    MicromechanicalModel,
    ThermoelasticDamageModel,
    PhaseFieldBrittleFracture,
)
from pyfoam.structural.stress_solver_enhanced_8 import (
    EnhancedStressSolver8,
    PhaseFieldFatigueResult,
    StressRecoveryResult,
    MultiPhysicsStressResult,
)
from pyfoam.structural.displacement_solver_enhanced_8 import (
    EnhancedDisplacementSolver8,
    LevelSetResult,
    MultiMaterialResult,
    ConstrainedTopologyResult,
)
from pyfoam.structural.elastic_model_enhanced_9 import (
    FunctionallyGradedModel,
    CoupledPoromechanicsModel,
    ElectroMechanicalModel,
)
from pyfoam.structural.stress_solver_enhanced_9 import (
    EnhancedStressSolver9,
    MultiScaleResult,
    ErrorEstimatorResult,
    KrigingResult,
)
from pyfoam.structural.displacement_solver_enhanced_9 import (
    EnhancedDisplacementSolver9,
    IsogeometricResult,
    MeshlessResult,
    RefinementResult9,
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
    # V3 enhanced
    "OrthotropicPlasticModel",
    "ViscoelasticMaxwellModel",
    "DamageModel",
    "EnhancedStressSolver3",
    "NonlinearStressResult",
    "EnhancedDisplacementSolver3",
    "LargeDeformationResult",
    # V4 enhanced
    "GursonDamageModel",
    "CrystalPlasticityModel",
    "PhaseFieldFractureModel",
    "EnhancedStressSolver4",
    "SmoothedStressResult",
    "ThermalCoupling",
    "EnhancedDisplacementSolver4",
    "ContactResult",
    "RefinementIndicator",
    # V5 enhanced
    "GradientPlasticityModel",
    "CoupledDamagePlasticityModel",
    "HyperelasticOgdenModel",
    "EnhancedStressSolver5",
    "FailureAssessment",
    "StressInvariants",
    "EnhancedDisplacementSolver5",
    "ModalResult",
    "NewmarkResult",
    "RayleighDamping",
    # V6 enhanced
    "ChabocheKinematicHardening",
    "JohnsonCookModel",
    "ConcreteDamagedPlasticityModel",
    "EnhancedStressSolver6",
    "CrackResult",
    "FatigueResult",
    "CreepResult",
    "EnhancedDisplacementSolver6",
    "BucklingResult",
    "ContactResult6",
    "GeometricNonlinearResult",
    # V7 enhanced
    "ThermomechanicalCouplingModel",
    "PorousElasticModel",
    "FatigueDamageModel",
    "EnhancedStressSolver7",
    "XFEMResult",
    "ThermalStressResult",
    "HomogenisationResult",
    "EnhancedDisplacementSolver7",
    "TopologyResult",
    "RefinementResult7",
    "SubstructureResult",
    # V8 enhanced
    "MicromechanicalModel",
    "ThermoelasticDamageModel",
    "PhaseFieldBrittleFracture",
    "EnhancedStressSolver8",
    "PhaseFieldFatigueResult",
    "StressRecoveryResult",
    "MultiPhysicsStressResult",
    "EnhancedDisplacementSolver8",
    "LevelSetResult",
    "MultiMaterialResult",
    "ConstrainedTopologyResult",
    # V9 enhanced
    "FunctionallyGradedModel",
    "CoupledPoromechanicsModel",
    "ElectroMechanicalModel",
    "EnhancedStressSolver9",
    "MultiScaleResult",
    "ErrorEstimatorResult",
    "KrigingResult",
    "EnhancedDisplacementSolver9",
    "IsogeometricResult",
    "MeshlessResult",
    "RefinementResult9",
]
