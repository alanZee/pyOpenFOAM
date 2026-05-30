"""
pyfoam.lagrangian — Lagrangian particle tracking framework.

Provides:

- :class:`Particle`       — single particle data class
- :class:`Cloud`          — base particle container
- :class:`KinematicCloud` — tracking with drag, gravity, wall bounce
- Force models: :class:`GravityForce`, :class:`DragForce`, :class:`LiftForce`
- Injectors:    :class:`PointInjector`, :class:`ConeInjector`,
  :class:`PatchInjector`, :class:`RandomInjector`
- Dispersion:   :class:`NoDispersion`, :class:`GradientDispersion`,
  :class:`StochasticDispersion`
- Collision:    :class:`NoCollision`, :class:`PairCollision`,
  :class:`SoftSphereModel`
- Evaporation:  :class:`NoEvaporation`, :class:`RanzMarshallEvaporation`
- Oxidation:    :class:`NoOxidation`, :class:`FieldOxidation`
- Breakup:      :class:`NoBreakup`, :class:`ReitzDiwakar`
- Wall interaction: :class:`ElasticBounce`, :class:`Stick`,
  :class:`SplashModel`
- MPPIC:       :class:`MPPICModel`, :class:`StandardMPPIC`,
  :class:`FrictionModel`, :class:`SchaefferFriction`
- Spray:       :class:`SprayModel`, :class:`BlobAtomization`,
  :class:`TABBreakup`
- Reacting:    :class:`ReactingModel`, :class:`SinglePhaseReacting`,
  :class:`MultiPhaseReacting`
"""

from pyfoam.lagrangian.particle import Particle
from pyfoam.lagrangian.cloud import Cloud, KinematicCloud
from pyfoam.lagrangian.forces import (
    ParticleForce,
    GravityForce,
    DragForce,
    LiftForce,
)
from pyfoam.lagrangian.injection import (
    Injector,
    PointInjector,
    ConeInjector,
    PatchInjector,
    RandomInjector,
)
from pyfoam.lagrangian.dispersion import (
    DispersionModel,
    NoDispersion,
    GradientDispersion,
    StochasticDispersion,
)
from pyfoam.lagrangian.collision import (
    CollisionModel,
    NoCollision,
    PairCollision,
    SoftSphereModel,
)
from pyfoam.lagrangian.evaporation import (
    EvaporationModel,
    NoEvaporation,
    RanzMarshallEvaporation,
)
from pyfoam.lagrangian.oxidation import (
    OxidationModel,
    NoOxidation,
    FieldOxidation,
)
from pyfoam.lagrangian.breakup import (
    BreakupModel,
    NoBreakup,
    ReitzDiwakar,
)
from pyfoam.lagrangian.wall_interaction import (
    WallInteractionModel,
    ElasticBounce,
    Stick,
    SplashModel,
)
from pyfoam.lagrangian.mppic_models import (
    MPPICModel,
    StandardMPPIC,
    FrictionModel,
    SchaefferFriction,
)
from pyfoam.lagrangian.spray_models import (
    SprayModel,
    BlobAtomization,
    TABBreakup,
)
from pyfoam.lagrangian.reacting_models import (
    ReactingModel,
    SinglePhaseReacting,
    MultiPhaseReacting,
)

# ---------- Enhanced model variants v2-v5 ----------

from pyfoam.lagrangian.injection_enhanced_2 import CloudInjector, FieldInjector
from pyfoam.lagrangian.injection_enhanced_3 import LagrangianMappingInjector, ManualInjectionRate
from pyfoam.lagrangian.injection_enhanced_4 import ThermoCloudInjector, ReactingCloudInjector
from pyfoam.lagrangian.injection_enhanced_5 import KinematicParcelInjector, SolidParticleInjector

from pyfoam.lagrangian.forces_enhanced_2 import VirtualMassForce, SaffmanMeiLift
from pyfoam.lagrangian.forces_enhanced_3 import ThermophoreticForce, BrownianMotionForce
from pyfoam.lagrangian.forces_enhanced_4 import PressureGradientForce, BuoyancyForce
from pyfoam.lagrangian.forces_enhanced_5 import MagnusForce, ParamagneticForce

from pyfoam.lagrangian.breakup_enhanced_2 import PilchErdman, ReitzKHRT
from pyfoam.lagrangian.breakup_enhanced_3 import TABBreakup, SHFBreakup
from pyfoam.lagrangian.breakup_enhanced_4 import ETABBreakup, SSDBreakup
from pyfoam.lagrangian.breakup_enhanced_5 import EnhancedTaylorAnalogy, KHRTBreakup

from pyfoam.lagrangian.collision_enhanced_2 import TrajectoryModel, ORourkeCollision
from pyfoam.lagrangian.collision_enhanced_3 import StochasticCollision, NoSeparation
from pyfoam.lagrangian.collision_enhanced_4 import SpringDashpot, PairCollisionWall
from pyfoam.lagrangian.collision_enhanced_5 import SubCycledCollision, CoulalogluCollision

from pyfoam.lagrangian.dispersion_enhanced_2 import GradientDispersionRNG, StochasticDispersionRNG
from pyfoam.lagrangian.dispersion_enhanced_3 import TurbulentDispersion, GradientDispersionB
from pyfoam.lagrangian.dispersion_enhanced_4 import BrownianDispersion, DispersionModelRAS
from pyfoam.lagrangian.dispersion_enhanced_5 import DispersionModelKE, InverseTimeScaleDispersion

from pyfoam.lagrangian.evaporation_enhanced_2 import StandardEvaporation, DiffusionEvaporation
from pyfoam.lagrangian.evaporation_enhanced_3 import LiquidEvaporation, MultiComponentEvaporation
from pyfoam.lagrangian.evaporation_enhanced_4 import BlowingEvaporation, NoEvaporationTwoPhase
from pyfoam.lagrangian.evaporation_enhanced_5 import EvaporationDI, FrosslingEvaporation

from pyfoam.lagrangian.mppic_enhanced_2 import SyamlalRogersFriction, GidaspowFriction
from pyfoam.lagrangian.mppic_enhanced_3 import ErgunFriction, PackingLimitModel
from pyfoam.lagrangian.mppic_enhanced_4 import IsotropicDamping, VelocityLimiter
from pyfoam.lagrangian.mppic_enhanced_5 import MinimumMassLimiter, MaximumTemperatureLimiter

from pyfoam.lagrangian.oxidation_enhanced_2 import IntrinsicOxidation, CKAOxidation
from pyfoam.lagrangian.oxidation_enhanced_3 import KineticDiffusionOxidation, DiffusionLimitedOxidation
from pyfoam.lagrangian.oxidation_enhanced_4 import LiquidEvaporationOxidation, SurfaceReaction
from pyfoam.lagrangian.oxidation_enhanced_5 import RandomPoreModel, ShrinkingCoreModel

from pyfoam.lagrangian.spray_enhanced_2 import KHRTAtomization, LISAAtomization
from pyfoam.lagrangian.spray_enhanced_3 import ReitzKHRTBreakup, SSDAtomization
from pyfoam.lagrangian.spray_enhanced_4 import BlobAtomizationNozzle, SprayPostProcessing
from pyfoam.lagrangian.spray_enhanced_5 import NozzleFlowModel, SheetAtomization

from pyfoam.lagrangian.wall_enhanced_2 import BaiGosmanSplash, KuhnkeSplash
from pyfoam.lagrangian.wall_enhanced_3 import ReboundModel, StochasticSplash
from pyfoam.lagrangian.wall_enhanced_4 import ThermalWallInteraction, WetWallInteraction
from pyfoam.lagrangian.wall_enhanced_5 import WallBounceDistribution, CriticalVelocityModel

from pyfoam.lagrangian.reacting_enhanced_2 import CompositionModel, PhaseChangeModel
from pyfoam.lagrangian.reacting_enhanced_3 import ReactingMultiphaseCloud, TwoPhaseReacting
from pyfoam.lagrangian.reacting_enhanced_4 import HeterogeneousReacting, CatalyticReacting
from pyfoam.lagrangian.reacting_enhanced_5 import DevolatilisationModel, CharBurnoutModel

# --- Enhanced variants v6-v10 ---
from pyfoam.lagrangian.injection_enhanced_6 import InjectionFromFile, ResettableInjector
from pyfoam.lagrangian.injection_enhanced_7 import SurfaceFluxInjector, VolumeSourceInjector
from pyfoam.lagrangian.injection_enhanced_8 import RateControlledInjector, DistributionInjector
from pyfoam.lagrangian.injection_enhanced_9 import TemporalInjector, ProbabilisticInjector
from pyfoam.lagrangian.injection_enhanced_10 import MultiPointInjector, AdaptiveInjector

from pyfoam.lagrangian.forces_enhanced_6 import ChargedParticleForce, BassetForce
from pyfoam.lagrangian.forces_enhanced_7 import FaxenForce, HistoryForce
from pyfoam.lagrangian.forces_enhanced_8 import OseenDragForce, HadamardRybczynskiDrag
from pyfoam.lagrangian.forces_enhanced_9 import CoriolisForce, CentrifugalForce
from pyfoam.lagrangian.forces_enhanced_10 import ElectrostaticForce, AcousticRadiationForce

from pyfoam.lagrangian.breakup_enhanced_6 import LISABreakup, SchmehlBreakup
from pyfoam.lagrangian.breakup_enhanced_7 import WAVEBreakup, MadabhushiBreakup
from pyfoam.lagrangian.breakup_enhanced_8 import CascadeBreakup, PowerLawBreakup
from pyfoam.lagrangian.breakup_enhanced_9 import FractalBreakup, RosinRammlerBreakup
from pyfoam.lagrangian.breakup_enhanced_10 import FragBreakup, UniformBreakup

from pyfoam.lagrangian.collision_enhanced_6 import DeterministicCollision, NTCollision
from pyfoam.lagrangian.collision_enhanced_7 import AdaptiveCollision, MultiParticleCollision
from pyfoam.lagrangian.collision_enhanced_8 import SphereCollision, CellCollision
from pyfoam.lagrangian.collision_enhanced_9 import ReversibleCollision, InelasticCoalescence
from pyfoam.lagrangian.collision_enhanced_10 import DEMCollision, PSCollision

from pyfoam.lagrangian.dispersion_enhanced_6 import LangevinDispersion, FilteredDispersion
from pyfoam.lagrangian.dispersion_enhanced_7 import TensorDispersion, SchmidtDispersion
from pyfoam.lagrangian.dispersion_enhanced_8 import EddyInteractionDispersion, CrossDispersion
from pyfoam.lagrangian.dispersion_enhanced_9 import DiffusionDispersion, DriftDispersion
from pyfoam.lagrangian.dispersion_enhanced_10 import FilteredDNSDispersion, ScaleSimilarityDispersion

from pyfoam.lagrangian.evaporation_enhanced_6 import SkinEvaporation, HomogeneousEvaporation
from pyfoam.lagrangian.evaporation_enhanced_7 import ShellEvaporation, KnudsenEvaporation
from pyfoam.lagrangian.evaporation_enhanced_8 import FuelEvaporation, FilmEvaporation
from pyfoam.lagrangian.evaporation_enhanced_9 import FlashEvaporation, ConvectiveEvaporation
from pyfoam.lagrangian.evaporation_enhanced_10 import EquilibriumEvaporation, NonEquilibriumEvaporation

from pyfoam.lagrangian.mppic_models_enhanced_6 import JohnsonJacksonFriction, KTGFStress
from pyfoam.lagrangian.mppic_models_enhanced_7 import LunSavageFriction, GranularTemperatureModel
from pyfoam.lagrangian.mppic_models_enhanced_8 import ImplicitDamping, ExplicitDamping
from pyfoam.lagrangian.mppic_models_enhanced_9 import MaximumVelocityLimiter, MinimumDiameterLimiter
from pyfoam.lagrangian.mppic_models_enhanced_10 import PackingLimiter, VolumeFractionSmooth

from pyfoam.lagrangian.oxidation_enhanced_6 import LangmuirHinshelwoodOxidation, MarsMaesoneOxidation
from pyfoam.lagrangian.oxidation_enhanced_7 import PowerLawOxidation, NthOrderOxidation
from pyfoam.lagrangian.oxidation_enhanced_8 import BidisperseOxidation, RandomPoreV2
from pyfoam.lagrangian.oxidation_enhanced_9 import GrainModel, VolumeReactionModel
from pyfoam.lagrangian.oxidation_enhanced_10 import KineticDiffusionV2, MultipleReactionOxidation

from pyfoam.lagrangian.spray_models_enhanced_6 import WaveAtomization, FIPAAtomization
from pyfoam.lagrangian.spray_models_enhanced_7 import BlobsheetAtomization, FilmAtomization
from pyfoam.lagrangian.spray_models_enhanced_8 import CascadeAtomization, StochasticAtomization
from pyfoam.lagrangian.spray_models_enhanced_9 import RTAtomization, MultimodeAtomization
from pyfoam.lagrangian.spray_models_enhanced_10 import HybridAtomization, CalibratedAtomization

from pyfoam.lagrangian.wall_interaction_enhanced_6 import MomentumTransferWall, HeatTransferWall
from pyfoam.lagrangian.wall_interaction_enhanced_7 import BounceFrictionWall, AbsorptionWall
from pyfoam.lagrangian.wall_interaction_enhanced_8 import SplashFragmentWall, SplashCoalescence
from pyfoam.lagrangian.wall_interaction_enhanced_9 import TemperatureDependentWall, MaterialPropertyWall
from pyfoam.lagrangian.wall_interaction_enhanced_10 import ProbabilisticWall, AdaptiveWall

from pyfoam.lagrangian.reacting_models_enhanced_6 import KineticReacting, DiffusionReacting
from pyfoam.lagrangian.reacting_models_enhanced_7 import ArrheniusReacting, EquilibriumReacting
from pyfoam.lagrangian.reacting_models_enhanced_8 import ShrinkingCoreReacting, UniformConversionReacting
from pyfoam.lagrangian.reacting_models_enhanced_9 import GlobalReactionModel, DetailedReactionModel
from pyfoam.lagrangian.reacting_models_enhanced_10 import CharGasificationModel, PyrolysisModel

__all__ = [
    "Particle",
    "Cloud",
    "KinematicCloud",
    "ParticleForce",
    "GravityForce",
    "DragForce",
    "LiftForce",
    "Injector",
    "PointInjector",
    "ConeInjector",
    "PatchInjector",
    "RandomInjector",
    "DispersionModel",
    "NoDispersion",
    "GradientDispersion",
    "StochasticDispersion",
    "CollisionModel",
    "NoCollision",
    "PairCollision",
    "SoftSphereModel",
    "EvaporationModel",
    "NoEvaporation",
    "RanzMarshallEvaporation",
    "OxidationModel",
    "NoOxidation",
    "FieldOxidation",
    "BreakupModel",
    "NoBreakup",
    "ReitzDiwakar",
    "WallInteractionModel",
    "ElasticBounce",
    "Stick",
    "SplashModel",
    # MPPIC models
    "MPPICModel",
    "StandardMPPIC",
    "FrictionModel",
    "SchaefferFriction",
    # Spray models
    "SprayModel",
    "BlobAtomization",
    "TABBreakup",
    # Reacting models
    "ReactingModel",
    "SinglePhaseReacting",
    "MultiPhaseReacting",
    # --- Enhanced variants v2-v5 ---
    # Injection enhanced
    "CloudInjector", "FieldInjector",
    "LagrangianMappingInjector", "ManualInjectionRate",
    "ThermoCloudInjector", "ReactingCloudInjector",
    "KinematicParcelInjector", "SolidParticleInjector",
    # Forces enhanced
    "VirtualMassForce", "SaffmanMeiLift",
    "ThermophoreticForce", "BrownianMotionForce",
    "PressureGradientForce", "BuoyancyForce",
    "MagnusForce", "ParamagneticForce",
    # Breakup enhanced
    "PilchErdman", "ReitzKHRT",
    "TABBreakup", "SHFBreakup",
    "ETABBreakup", "SSDBreakup",
    "EnhancedTaylorAnalogy", "KHRTBreakup",
    # Collision enhanced
    "TrajectoryModel", "ORourkeCollision",
    "StochasticCollision", "NoSeparation",
    "SpringDashpot", "PairCollisionWall",
    "SubCycledCollision", "CoulalogluCollision",
    # Dispersion enhanced
    "GradientDispersionRNG", "StochasticDispersionRNG",
    "TurbulentDispersion", "GradientDispersionB",
    "BrownianDispersion", "DispersionModelRAS",
    "DispersionModelKE", "InverseTimeScaleDispersion",
    # Evaporation enhanced
    "StandardEvaporation", "DiffusionEvaporation",
    "LiquidEvaporation", "MultiComponentEvaporation",
    "BlowingEvaporation", "NoEvaporationTwoPhase",
    "EvaporationDI", "FrosslingEvaporation",
    # MPPIC enhanced
    "SyamlalRogersFriction", "GidaspowFriction",
    "ErgunFriction", "PackingLimitModel",
    "IsotropicDamping", "VelocityLimiter",
    "MinimumMassLimiter", "MaximumTemperatureLimiter",
    # Oxidation enhanced
    "IntrinsicOxidation", "CKAOxidation",
    "KineticDiffusionOxidation", "DiffusionLimitedOxidation",
    "LiquidEvaporationOxidation", "SurfaceReaction",
    "RandomPoreModel", "ShrinkingCoreModel",
    # Spray enhanced
    "KHRTAtomization", "LISAAtomization",
    "ReitzKHRTBreakup", "SSDAtomization",
    "BlobAtomizationNozzle", "SprayPostProcessing",
    "NozzleFlowModel", "SheetAtomization",
    # Wall interaction enhanced
    "BaiGosmanSplash", "KuhnkeSplash",
    "ReboundModel", "StochasticSplash",
    "ThermalWallInteraction", "WetWallInteraction",
    "WallBounceDistribution", "CriticalVelocityModel",
    # Reacting enhanced
    "CompositionModel", "PhaseChangeModel",
    "ReactingMultiphaseCloud", "TwoPhaseReacting",
    "HeterogeneousReacting", "CatalyticReacting",
    "DevolatilisationModel", "CharBurnoutModel",
    # --- Enhanced variants v6-v10 ---
    "InjectionFromFile", "ResettableInjector",
    "SurfaceFluxInjector", "VolumeSourceInjector",
    "RateControlledInjector", "DistributionInjector",
    "TemporalInjector", "ProbabilisticInjector",
    "MultiPointInjector", "AdaptiveInjector",
    "ChargedParticleForce", "BassetForce",
    "FaxenForce", "HistoryForce",
    "OseenDragForce", "HadamardRybczynskiDrag",
    "CoriolisForce", "CentrifugalForce",
    "ElectrostaticForce", "AcousticRadiationForce",
    "LISABreakup", "SchmehlBreakup",
    "WAVEBreakup", "MadabhushiBreakup",
    "CascadeBreakup", "PowerLawBreakup",
    "FractalBreakup", "RosinRammlerBreakup",
    "FragBreakup", "UniformBreakup",
    "DeterministicCollision", "NTCollision",
    "AdaptiveCollision", "MultiParticleCollision",
    "SphereCollision", "CellCollision",
    "ReversibleCollision", "InelasticCoalescence",
    "DEMCollision", "PSCollision",
    "LangevinDispersion", "FilteredDispersion",
    "TensorDispersion", "SchmidtDispersion",
    "EddyInteractionDispersion", "CrossDispersion",
    "DiffusionDispersion", "DriftDispersion",
    "FilteredDNSDispersion", "ScaleSimilarityDispersion",
    "SkinEvaporation", "HomogeneousEvaporation",
    "ShellEvaporation", "KnudsenEvaporation",
    "FuelEvaporation", "FilmEvaporation",
    "FlashEvaporation", "ConvectiveEvaporation",
    "EquilibriumEvaporation", "NonEquilibriumEvaporation",
    "JohnsonJacksonFriction", "KTGFStress",
    "LunSavageFriction", "GranularTemperatureModel",
    "ImplicitDamping", "ExplicitDamping",
    "MaximumVelocityLimiter", "MinimumDiameterLimiter",
    "PackingLimiter", "VolumeFractionSmooth",
    "LangmuirHinshelwoodOxidation", "MarsMaesoneOxidation",
    "PowerLawOxidation", "NthOrderOxidation",
    "BidisperseOxidation", "RandomPoreV2",
    "GrainModel", "VolumeReactionModel",
    "KineticDiffusionV2", "MultipleReactionOxidation",
    "WaveAtomization", "FIPAAtomization",
    "BlobsheetAtomization", "FilmAtomization",
    "CascadeAtomization", "StochasticAtomization",
    "RTAtomization", "MultimodeAtomization",
    "HybridAtomization", "CalibratedAtomization",
    "MomentumTransferWall", "HeatTransferWall",
    "BounceFrictionWall", "AbsorptionWall",
    "SplashFragmentWall", "SplashCoalescence",
    "TemperatureDependentWall", "MaterialPropertyWall",
    "ProbabilisticWall", "AdaptiveWall",
    "KineticReacting", "DiffusionReacting",
    "ArrheniusReacting", "EquilibriumReacting",
    "ShrinkingCoreReacting", "UniformConversionReacting",
    "GlobalReactionModel", "DetailedReactionModel",
    "CharGasificationModel", "PyrolysisModel",
]
