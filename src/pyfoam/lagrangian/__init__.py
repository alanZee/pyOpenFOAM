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
]
