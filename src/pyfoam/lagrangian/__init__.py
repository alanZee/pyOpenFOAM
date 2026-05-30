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
]
