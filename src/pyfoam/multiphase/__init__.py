"""
pyfoam.multiphase — Multiphase flow models.

Provides:

- :class:`VOFAdvection` — Volume of Fluid advection with interface compression
- :class:`MULESLimiter` — Bounded scalar transport limiter
- :class:`SurfaceTensionModel` — Continuum Surface Force (CSF) model
- Interphase models: SchillerNaumannDrag, WenYuDrag, GidaspowDrag,
  TomiyamaLift, VirtualMassForce
- Drag model ABC: :class:`DragModel` with RTS registry
  (``pyfoam.multiphase.drag_models``)
- Lift model ABC: :class:`LiftModel` with RTS registry
  (``pyfoam.multiphase.lift_models``)
- :class:`SaffmanLift` — Saffman shear-induced lift force
- Cavitation models: SchnerrSauer, Merkle, ZGB
- Enhanced cavitation models: ZGBModel, MerkleModel (with convergence enhancements)
- Interface reconstruction: PLICReconstruction
- :class:`PopulationBalanceModel` — Population balance equation solver
  (method of classes, droplet/bubble size distribution tracking)
- :class:`BubbleModel` — Abstract bubble diameter model with RTS registry
- :class:`ConstantBubble` — Constant (fixed) bubble diameter
- :class:`BubbleBreakup` — Bubble breakup/coalescence equilibrium model
- :class:`PhaseChangeModel` — Abstract phase change model with RTS registry
- :class:`LeeModel` — Lee empirical phase change model
- :class:`SchnerrSauerEnhanced` — Enhanced Schnerr-Sauer with convergence improvements
- :class:`TurbulenceDampingModel` — Abstract turbulence damping model with RTS registry
- :class:`InterfaceDamping` — Damps k and epsilon near free surface
- :class:`InterfacialAreaModel` — Abstract interfacial area density model with RTS registry
- :class:`ConstantInterfacialArea` — Constant interfacial area density
- :class:`VariableInterfacialArea` — Alpha-dependent interfacial area density
- :class:`RelativeVelocityModel` — Abstract base for relative velocity models
- :class:`ManninenRelativeVelocity` — Manninen et al. algebraic slip model
- :class:`GraceRelativeVelocity` — Grace drag correlation for bubbles/particles
- :class:`TurbulenceInteractionModel` — Abstract interphase turbulence interaction model
- :class:`StandardInteraction` — Standard interphase turbulence interaction (Lopez de Bertodano)
- :class:`WallLubricationModel` — Abstract wall lubrication force model with RTS registry
- :class:`AntalWallLubrication` — Antal et al. (1991) distance-dependent wall lubrication
- :class:`TomiyamaWallLubrication` — Tomiyama et al. (1998) Eo-dependent wall lubrication
- :class:`TurbulenceDamping2Model` — Enhanced turbulence damping with y+ awareness
- :class:`WolfhardtDamping` — Wolfhardt model for near-wall turbulence damping
- :class:`TurbulenceWallDampingModel` — Abstract wall damping model for VOF with RTS registry
- :class:`BrackbillDamping` — Brackbill near-wall damping for VOF simulations
- :class:`VirtualMassModel` — Abstract virtual mass force model with RTS registry
- :class:`ConstantVirtualMass` — Constant virtual mass coefficient
- :class:`LambVirtualMass` — Lamb's inviscid virtual mass (C_vm = 0.5)
- :class:`KatoLaunderDamping2` — Enhanced Kato-Launder damping with alpha-dependent and gradient damping
- :class:`IncompressibleDriftFlux` — Incompressible drift-flux mixture model
- :class:`DenseParticleFluid` — Dense particle-laden Eulerian flow model
- :class:`FilmModel` — Thin film flow model
- :class:`XiFluid` — B-Xi two-equation premixed combustion model
- :class:`PLICReconstructionStandalone` — Enhanced standalone PLIC (mesh-free)
- Contact angle models: ContactAngleModel, ConstantContactAngle,
  DynamicContactAngle, KistlerContactAngle
- Enhanced PBE solvers: MOMSolver, QMOMSolver, SectionalSolver
- Enhanced turbulence damping v2: TurbulenceDamping2EnhancedModel,
  LopezDeBertodanoDamping, KataokaDamping
- Enhanced incompressible N-phase VOF v2: IncompressibleMultiphaseVoFEnhanced
- Enhanced compressible N-phase VOF v2: CompressibleMultiphaseVoFEnhanced
- Enhanced multicomponent mixture v2: MulticomponentMixtureEnhanced
- Enhanced interfacial area v2: InterfacialArea2Model,
  DiameterTransportArea, LuoCoalescenceBreakupArea
- Enhanced turbulence damping v3: TurbulenceDamping3EnhancedModel,
  WallDampedDamping, SpaldingDamping, BlendedWallInterfaceDamping
- Enhanced incompressible N-phase VOF v4: IncompressibleMultiphaseVoFEnhanced3
  (curvature correction, hierarchical clamp, slip correction)
- Enhanced compressible N-phase VOF v4: CompressibleMultiphaseVoFEnhanced3
  (relaxed EOS, mixing correction, transonic limiter)
- Enhanced multicomponent mixture v4: MulticomponentMixtureEnhanced3
  (reactive transport, Wassiljewa conductivity, Lewis number correction)
- Enhanced interfacial area v4: InterfacialArea4Model,
  WaveBreakupArea, StretchRateArea, UnifiedBreakupCoalescenceArea
- Enhanced turbulence damping v5: TurbulenceDamping5EnhancedModel,
  NearWallAnisotropicDamping, BetaDampedModel, MultiScaleDamping
- Enhanced incompressible N-phase VOF v5: IncompressibleMultiphaseVoFEnhanced4
  (gradient-adaptive compression, normal smoothing, multi-pass bounded sweep)
- Enhanced compressible N-phase VOF v5: CompressibleMultiphaseVoFEnhanced4
  (EOS consistency check, enhanced PISO, Wood speed of sound)
- Enhanced multicomponent mixture v5: MulticomponentMixtureEnhanced4
  (Soret/Dufour coupling, Wilke viscosity mixing)
- Enhanced interfacial area v6: InterfacialArea5Model,
  TopologicalInterfaceArea, FractalDimensionArea, PhaseAwareAreaTransport
- Enhanced turbulence damping v6: TurbulenceDamping6EnhancedModel,
  LESAwareDamping, DynamicCoefficientDamping, TopologyAwareDamping
- Enhanced incompressible N-phase VOF v6: IncompressibleMultiphaseVoFEnhanced5
  (surface tension reconstruction, adaptive interface refinement, phase-aware Courant)
- Enhanced compressible N-phase VOF v6: CompressibleMultiphaseVoFEnhanced5
  (energy coupling, non-equilibrium phase change thermal coupling)
- Enhanced multicomponent mixture v6: MulticomponentMixtureEnhanced5
  (Stefan-Maxwell diffusion, NRTL activity coefficients, mixture enthalpy)
- Enhanced interfacial area v7: InterfacialArea6Model,
  StochasticInterfacialArea, WeberCorrectedArea, NucleationAreaGeneration
- Enhanced turbulence damping v7: TurbulenceDamping7EnhancedModel,
  MLAssistedDamping, AnisotropicTensorDamping, ShearLayerDamping
"""

from pyfoam.multiphase.volume_of_fluid import VOFAdvection
from pyfoam.multiphase.mules import MULESLimiter
from pyfoam.multiphase.surface_tension import SurfaceTensionModel
from pyfoam.multiphase.interphase_models import (
    SchillerNaumannDrag,
    WenYuDrag,
    GidaspowDrag,
    TomiyamaLift,
    VirtualMassForce,
)
from pyfoam.multiphase.cavitation import SchnerrSauer, Merkle, ZGB
from pyfoam.multiphase.interface_reconstruction import (
    InterfaceReconstruction,
    PLICReconstruction,
)
from pyfoam.multiphase.interface_compression import InterfaceCompression
from pyfoam.multiphase.surface_tension_2 import CSFSurfaceTension
from pyfoam.multiphase.population_balance import (
    PopulationBalanceModel,
    PBEBin,
    ConstantCoalescence,
    ShearCoalescence,
    ConstantBreakup,
    WeberBreakup,
    ShearBreakup,
)
from pyfoam.multiphase.cavitation_models_enhanced import ZGBModel, MerkleModel
from pyfoam.multiphase.bubble_models import (
    BubbleModel,
    ConstantBubble,
    BubbleBreakup,
)
from pyfoam.multiphase.drift_flux_models import (
    DriftFluxModel,
    SimpleDriftFlux,
    GeneralDriftFlux,
)
from pyfoam.multiphase.phase_change import (
    PhaseChangeModel,
    LeeModel,
    SchnerrSauerEnhanced,
)
from pyfoam.multiphase.turbulence_damping import (
    TurbulenceDampingModel,
    InterfaceDamping,
)
from pyfoam.multiphase.mass_transfer import (
    MassTransferModel,
    LeeMassTransfer,
    ThermalPhaseChange,
)
from pyfoam.multiphase.interfacial_area import (
    InterfacialAreaModel,
    ConstantInterfacialArea,
    VariableInterfacialArea,
)
from pyfoam.multiphase.relative_velocity import (
    RelativeVelocityModel,
    ManninenRelativeVelocity,
    GraceRelativeVelocity,
)

# Phase 7: Drag and lift model ABC hierarchies
from pyfoam.multiphase.drag_models import DragModel
from pyfoam.multiphase.lift_models import LiftModel, SaffmanLift

# Phase 7: Turbulence transfer models
from pyfoam.multiphase.turbulence_transfer import (
    TurbulenceTransferModel,
    ContinuousTurbulenceTransfer,
    DispersedTurbulenceTransfer,
)

# Phase 7: Turbulence interaction models
from pyfoam.multiphase.turbulence_interaction import (
    TurbulenceInteractionModel,
    StandardInteraction,
)

# Phase 7: Wall lubrication models
from pyfoam.multiphase.wall_lubrication_models import (
    WallLubricationModel,
    AntalWallLubrication,
    TomiyamaWallLubrication,
)

# Phase 7: Enhanced turbulence damping models
from pyfoam.multiphase.turbulence_damping_2 import (
    TurbulenceDamping2Model,
    WolfhardtDamping,
)

# Phase 7: Wall damping models for VOF
from pyfoam.multiphase.turbulence_wall_damping import (
    TurbulenceWallDampingModel,
    BrackbillDamping,
)

# Phase 7: Virtual mass models
from pyfoam.multiphase.virtual_mass_models import (
    VirtualMassModel,
    ConstantVirtualMass,
    LambVirtualMass,
)

# Phase 7: Enhanced Kato-Launder damping for multiphase
from pyfoam.multiphase.turbulence_kato_launder_2 import KatoLaunderDamping2

# Phase 8: Incompressible drift-flux mixture model
from pyfoam.multiphase.incompressible_drift_flux import IncompressibleDriftFlux

# Phase 8: Dense particle-laden flow model
from pyfoam.multiphase.dense_particle_fluid import DenseParticleFluid

# Phase 8: Thin film flow model
from pyfoam.multiphase.film_model import FilmModel

# Phase 8: Xi premixed combustion model
from pyfoam.multiphase.xi_fluid import XiFluid

# Phase 8: Enhanced PLIC interface reconstruction (standalone)
from pyfoam.multiphase.pllic import PLICReconstruction as PLICReconstructionStandalone

# Phase 8: Contact angle models
from pyfoam.multiphase.contact_angle_models import (
    ContactAngleModel,
    ConstantContactAngle,
    DynamicContactAngle,
    KistlerContactAngle,
)

# Phase 9: Incompressible N-phase VOF
from pyfoam.multiphase.incompressible_multiphase_vof import IncompressibleMultiphaseVoF

# Phase 9: Compressible N-phase VOF
from pyfoam.multiphase.compressible_multiphase_vof import CompressibleMultiphaseVoF

# Phase 9: Multi-component mixture
from pyfoam.multiphase.multicomponent_mixture import MulticomponentMixture

# Phase 9: Enhanced interfacial area models
from pyfoam.multiphase.interfacial_area_enhanced import (
    SauterMeanInterfacialArea,
    BreakupCoalescenceInterfacialArea,
    BlendedInterfacialArea,
)

# Phase 9: Enhanced turbulence damping
from pyfoam.multiphase.turbulence_damping_enhanced import (
    TurbulenceDampingEnhancedModel,
    GradientDamping,
    ExponentialBlendedDamping,
)

# Phase 10: Enhanced population balance solvers
from pyfoam.multiphase.population_balance_enhanced import (
    MOMSolver,
    QMOMSolver,
    SectionalSolver,
)

# Phase 10: Enhanced turbulence damping — Phase 2
from pyfoam.multiphase.turbulence_damping_enhanced_2 import (
    TurbulenceDamping2EnhancedModel,
    LopezDeBertodanoDamping,
    KataokaDamping,
)

# Phase 11: Enhanced incompressible N-phase VOF v2
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced import (
    IncompressibleMultiphaseVoFEnhanced,
)

# Phase 11: Enhanced compressible N-phase VOF v2
from pyfoam.multiphase.compressible_multiphase_vof_enhanced import (
    CompressibleMultiphaseVoFEnhanced,
)

# Phase 11: Enhanced multicomponent mixture v2
from pyfoam.multiphase.multicomponent_mixture_enhanced import (
    MulticomponentMixtureEnhanced,
)

# Phase 11: Enhanced interfacial area v2
from pyfoam.multiphase.interfacial_area_enhanced_2 import (
    InterfacialArea2Model,
    DiameterTransportArea,
    LuoCoalescenceBreakupArea,
)

# Phase 11: Enhanced turbulence damping v3 (near-wall)
from pyfoam.multiphase.turbulence_damping_enhanced_3 import (
    TurbulenceDamping3EnhancedModel,
    WallDampedDamping,
    SpaldingDamping,
    BlendedWallInterfaceDamping,
)

# Phase 12: Enhanced incompressible N-phase VOF v3
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_2 import (
    IncompressibleMultiphaseVoFEnhanced2,
)

# Phase 12: Enhanced compressible N-phase VOF v3
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_2 import (
    CompressibleMultiphaseVoFEnhanced2,
)

# Phase 12: Enhanced multicomponent mixture v3
from pyfoam.multiphase.multicomponent_mixture_enhanced_2 import (
    MulticomponentMixtureEnhanced2,
)

# Phase 12: Enhanced interfacial area v3
from pyfoam.multiphase.interfacial_area_enhanced_3 import (
    InterfacialArea3Model,
    TurbulentBreakupArea,
    TipStreamingArea,
)

# Phase 12: Enhanced turbulence damping v4
from pyfoam.multiphase.turbulence_damping_enhanced_4 import (
    TurbulenceDamping4EnhancedModel,
    ReynoldsAdaptiveDamping,
    TwoLayerDamping,
    AlphaGradientLimiter,
)

# Phase 13: Enhanced incompressible N-phase VOF v4
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_3 import (
    IncompressibleMultiphaseVoFEnhanced3,
)

# Phase 13: Enhanced compressible N-phase VOF v4
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_3 import (
    CompressibleMultiphaseVoFEnhanced3,
)

# Phase 13: Enhanced multicomponent mixture v4
from pyfoam.multiphase.multicomponent_mixture_enhanced_3 import (
    MulticomponentMixtureEnhanced3,
)

# Phase 13: Enhanced interfacial area v4
from pyfoam.multiphase.interfacial_area_enhanced_4 import (
    InterfacialArea4Model,
    WaveBreakupArea,
    StretchRateArea,
    UnifiedBreakupCoalescenceArea,
)

# Phase 13: Enhanced turbulence damping v5
from pyfoam.multiphase.turbulence_damping_enhanced_5 import (
    TurbulenceDamping5EnhancedModel,
    NearWallAnisotropicDamping,
    BetaDampedModel,
    MultiScaleDamping,
)

# Phase 14: Enhanced incompressible N-phase VOF v5
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_4 import (
    IncompressibleMultiphaseVoFEnhanced4,
)

# Phase 14: Enhanced compressible N-phase VOF v5
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_4 import (
    CompressibleMultiphaseVoFEnhanced4,
)

# Phase 14: Enhanced multicomponent mixture v5
from pyfoam.multiphase.multicomponent_mixture_enhanced_4 import (
    MulticomponentMixtureEnhanced4,
)

# Phase 14: Enhanced interfacial area v6
from pyfoam.multiphase.interfacial_area_enhanced_5 import (
    InterfacialArea5Model,
    TopologicalInterfaceArea,
    FractalDimensionArea,
    PhaseAwareAreaTransport,
)

# Phase 14: Enhanced turbulence damping v6
from pyfoam.multiphase.turbulence_damping_enhanced_6 import (
    TurbulenceDamping6EnhancedModel,
    LESAwareDamping,
    DynamicCoefficientDamping,
    TopologyAwareDamping,
)

# Phase 15: Enhanced incompressible N-phase VOF v6
from pyfoam.multiphase.incompressible_multiphase_vof_enhanced_5 import (
    IncompressibleMultiphaseVoFEnhanced5,
)

# Phase 15: Enhanced compressible N-phase VOF v6
from pyfoam.multiphase.compressible_multiphase_vof_enhanced_5 import (
    CompressibleMultiphaseVoFEnhanced5,
)

# Phase 15: Enhanced multicomponent mixture v6
from pyfoam.multiphase.multicomponent_mixture_enhanced_5 import (
    MulticomponentMixtureEnhanced5,
)

# Phase 15: Enhanced interfacial area v7
from pyfoam.multiphase.interfacial_area_enhanced_6 import (
    InterfacialArea6Model,
    StochasticInterfacialArea,
    WeberCorrectedArea,
    NucleationAreaGeneration,
)

# Phase 15: Enhanced turbulence damping v7
from pyfoam.multiphase.turbulence_damping_enhanced_7 import (
    TurbulenceDamping7EnhancedModel,
    MLAssistedDamping,
    AnisotropicTensorDamping,
    ShearLayerDamping,
)

__all__ = [
    "VOFAdvection",
    "MULESLimiter",
    "SurfaceTensionModel",
    "SchillerNaumannDrag",
    "WenYuDrag",
    "GidaspowDrag",
    "TomiyamaLift",
    "VirtualMassForce",
    "SchnerrSauer",
    "Merkle",
    "ZGB",
    "InterfaceReconstruction",
    "PLICReconstruction",
    "InterfaceCompression",
    "CSFSurfaceTension",
    "PopulationBalanceModel",
    "PBEBin",
    "ConstantCoalescence",
    "ShearCoalescence",
    "ConstantBreakup",
    "WeberBreakup",
    "ShearBreakup",
    # Enhanced cavitation models
    "ZGBModel",
    "MerkleModel",
    # Bubble models
    "BubbleModel",
    "ConstantBubble",
    "BubbleBreakup",
    # Drift-flux models
    "DriftFluxModel",
    "SimpleDriftFlux",
    "GeneralDriftFlux",
    # Phase change models
    "PhaseChangeModel",
    "LeeModel",
    "SchnerrSauerEnhanced",
    # Turbulence damping models
    "TurbulenceDampingModel",
    "InterfaceDamping",
    # Mass transfer models
    "MassTransferModel",
    "LeeMassTransfer",
    "ThermalPhaseChange",
    # Interfacial area models
    "InterfacialAreaModel",
    "ConstantInterfacialArea",
    "VariableInterfacialArea",
    # Relative velocity models
    "RelativeVelocityModel",
    "ManninenRelativeVelocity",
    "GraceRelativeVelocity",
    # Phase 7: Drag and lift model ABCs
    "DragModel",
    "LiftModel",
    "SaffmanLift",
    # Phase 7: Turbulence transfer models
    "TurbulenceTransferModel",
    "ContinuousTurbulenceTransfer",
    "DispersedTurbulenceTransfer",
    # Phase 7: Turbulence interaction models
    "TurbulenceInteractionModel",
    "StandardInteraction",
    # Phase 7: Wall lubrication models
    "WallLubricationModel",
    "AntalWallLubrication",
    "TomiyamaWallLubrication",
    # Phase 7: Enhanced turbulence damping models
    "TurbulenceDamping2Model",
    "WolfhardtDamping",
    # Phase 7: Wall damping models for VOF
    "TurbulenceWallDampingModel",
    "BrackbillDamping",
    # Phase 7: Virtual mass models
    "VirtualMassModel",
    "ConstantVirtualMass",
    "LambVirtualMass",
    # Phase 7: Enhanced Kato-Launder damping
    "KatoLaunderDamping2",
    # Phase 8: Incompressible drift-flux mixture model
    "IncompressibleDriftFlux",
    # Phase 8: Dense particle-laden flow model
    "DenseParticleFluid",
    # Phase 8: Thin film flow model
    "FilmModel",
    # Phase 8: Xi premixed combustion model
    "XiFluid",
    # Phase 8: Enhanced PLIC (standalone)
    "PLICReconstructionStandalone",
    # Phase 8: Contact angle models
    "ContactAngleModel",
    "ConstantContactAngle",
    "DynamicContactAngle",
    "KistlerContactAngle",
    # Phase 9: N-phase VOF models
    "IncompressibleMultiphaseVoF",
    "CompressibleMultiphaseVoF",
    # Phase 9: Multi-component mixture
    "MulticomponentMixture",
    # Phase 9: Enhanced interfacial area models
    "SauterMeanInterfacialArea",
    "BreakupCoalescenceInterfacialArea",
    "BlendedInterfacialArea",
    # Phase 9: Enhanced turbulence damping
    "TurbulenceDampingEnhancedModel",
    "GradientDamping",
    "ExponentialBlendedDamping",
    # Phase 10: Enhanced population balance solvers
    "MOMSolver",
    "QMOMSolver",
    "SectionalSolver",
    # Phase 10: Enhanced turbulence damping — Phase 2
    "TurbulenceDamping2EnhancedModel",
    "LopezDeBertodanoDamping",
    "KataokaDamping",
    # Phase 11: Enhanced incompressible N-phase VOF v2
    "IncompressibleMultiphaseVoFEnhanced",
    # Phase 11: Enhanced compressible N-phase VOF v2
    "CompressibleMultiphaseVoFEnhanced",
    # Phase 11: Enhanced multicomponent mixture v2
    "MulticomponentMixtureEnhanced",
    # Phase 11: Enhanced interfacial area v2
    "InterfacialArea2Model",
    "DiameterTransportArea",
    "LuoCoalescenceBreakupArea",
    # Phase 11: Enhanced turbulence damping v3
    "TurbulenceDamping3EnhancedModel",
    "WallDampedDamping",
    "SpaldingDamping",
    "BlendedWallInterfaceDamping",
    # Phase 12: Enhanced incompressible N-phase VOF v3
    "IncompressibleMultiphaseVoFEnhanced2",
    # Phase 12: Enhanced compressible N-phase VOF v3
    "CompressibleMultiphaseVoFEnhanced2",
    # Phase 12: Enhanced multicomponent mixture v3
    "MulticomponentMixtureEnhanced2",
    # Phase 12: Enhanced interfacial area v3
    "InterfacialArea3Model",
    "TurbulentBreakupArea",
    "TipStreamingArea",
    # Phase 12: Enhanced turbulence damping v4
    "TurbulenceDamping4EnhancedModel",
    "ReynoldsAdaptiveDamping",
    "TwoLayerDamping",
    "AlphaGradientLimiter",
    # Phase 13: Enhanced incompressible N-phase VOF v4
    "IncompressibleMultiphaseVoFEnhanced3",
    # Phase 13: Enhanced compressible N-phase VOF v4
    "CompressibleMultiphaseVoFEnhanced3",
    # Phase 13: Enhanced multicomponent mixture v4
    "MulticomponentMixtureEnhanced3",
    # Phase 13: Enhanced interfacial area v4
    "InterfacialArea4Model",
    "WaveBreakupArea",
    "StretchRateArea",
    "UnifiedBreakupCoalescenceArea",
    # Phase 13: Enhanced turbulence damping v5
    "TurbulenceDamping5EnhancedModel",
    "NearWallAnisotropicDamping",
    "BetaDampedModel",
    "MultiScaleDamping",
    # Phase 14: Enhanced incompressible N-phase VOF v5
    "IncompressibleMultiphaseVoFEnhanced4",
    # Phase 14: Enhanced compressible N-phase VOF v5
    "CompressibleMultiphaseVoFEnhanced4",
    # Phase 14: Enhanced multicomponent mixture v5
    "MulticomponentMixtureEnhanced4",
    # Phase 14: Enhanced interfacial area v6
    "InterfacialArea5Model",
    "TopologicalInterfaceArea",
    "FractalDimensionArea",
    "PhaseAwareAreaTransport",
    # Phase 14: Enhanced turbulence damping v6
    "TurbulenceDamping6EnhancedModel",
    "LESAwareDamping",
    "DynamicCoefficientDamping",
    "TopologyAwareDamping",
    # Phase 15: Enhanced incompressible N-phase VOF v6
    "IncompressibleMultiphaseVoFEnhanced5",
    # Phase 15: Enhanced compressible N-phase VOF v6
    "CompressibleMultiphaseVoFEnhanced5",
    # Phase 15: Enhanced multicomponent mixture v6
    "MulticomponentMixtureEnhanced5",
    # Phase 15: Enhanced interfacial area v7
    "InterfacialArea6Model",
    "StochasticInterfacialArea",
    "WeberCorrectedArea",
    "NucleationAreaGeneration",
    # Phase 15: Enhanced turbulence damping v7
    "TurbulenceDamping7EnhancedModel",
    "MLAssistedDamping",
    "AnisotropicTensorDamping",
    "ShearLayerDamping",
]
