"""
pyfoam.fv — Finite volume constraint & source model framework.

Provides:

**fvConstraints** (post-solution field modifications):

- :class:`FvConstraint` — abstract base with RTS registry and ``apply(field)`` interface
- :class:`BoundConstraint` — clamp field values to [min, max]
- :class:`FixedValueConstraint` — fix values at specified cell indices
- :class:`LimitPressureConstraint` — enforce non-negative pressure
- :class:`LimitTemperatureConstraint` — enforce physical temperature range
- :class:`MinMaxConstraint` — independent min/max with cell restriction
- :class:`RhoLimitsConstraint` — density bounds enforcement
- :class:`VelocityLimitsConstraint` — velocity magnitude limits

**fvModels** (pre-solution source term injection):

- :class:`FvModel` — abstract base with RTS registry and ``apply(matrix, field)`` interface
- :class:`SemiImplicitSource` — generic ``Su + Sp * phi`` volumetric source
- :class:`MassSource` — mass source / sink in continuity
- :class:`HeatSource` — volumetric heat source in energy equation
- :class:`PorosityForce` — Darcy-Forchheimer porosity resistance
- :class:`CodedFvModel` — user-defined Python function as source
- :class:`ActuationDiskModel` — actuation disk thrust source for wind turbines
- :class:`HeatExchangerModel` — temperature-dependent heat exchanger source
- :class:`SolidificationMeltingModel` — phase change in solids
- :class:`RASourceModel` — radiation absorption heat source
- :class:`GravitationalBodyForce` — gravitational body force
- :class:`CellSetSemiImplicitSource` — cellSet-restricted semi-implicit source
- :class:`PatchSemiImplicitSource` — patch-adjacent semi-implicit source
- :class:`ExplicitSource` — purely explicit volumetric source
- :class:`BuoyancyForce` — buoyancy force (rho - rho_ref) * g
- :class:`BoussinesqBuoyancy` — Boussinesq approximation buoyancy
- :class:`SRFForce` — single reference frame Coriolis + centrifugal force
- :class:`FvDOMRadiationSource` — FvDOM radiation source
- :class:`SolarLoadSource` — solar radiation heat load
- :class:`InterPhaseChangeModel` — cavitation/boiling interphase mass transfer

Usage::

    from pyfoam.fv import FvConstraint, BoundConstraint
    from pyfoam.fv import FvModel, SemiImplicitSource

    # fvConstraints: applied after solver step
    constraint = FvConstraint.create("bound", min=0.0, max=1.0)
    constraint.apply(field)

    # fvModels: applied before solver step
    model = FvModel.create("semiImplicitSource", Su=100.0, Sp=-0.5)
    model.apply(matrix, field)
"""

from pyfoam.fv.fv_constraints import (
    FvConstraint,
    BoundConstraint,
    FixedValueConstraint,
    LimitPressureConstraint,
    LimitTemperatureConstraint,
    create_constraint,
)

from pyfoam.fv.fv_models import (
    FvModel,
    SemiImplicitSource,
    MassSource,
    HeatSource,
    PorosityForce,
    CodedFvModel,
    create_fv_model,
)

from pyfoam.fv.actuation_disk import ActuationDiskModel
from pyfoam.fv.heat_exchanger import HeatExchangerModel
from pyfoam.fv.fv_sources import (
    SolidificationMeltingModel,
    RASourceModel,
    GravitationalBodyForce,
)

from pyfoam.fv.enhanced_2 import (
    CellSetSemiImplicitSource,
    PatchSemiImplicitSource,
    ExplicitSource,
)

from pyfoam.fv.enhanced_3 import (
    BuoyancyForce,
    BoussinesqBuoyancy,
    SRFForce,
)

from pyfoam.fv.enhanced_4 import (
    MinMaxConstraint,
    RhoLimitsConstraint,
    VelocityLimitsConstraint,
)

from pyfoam.fv.enhanced_5 import (
    FvDOMRadiationSource,
    SolarLoadSource,
    InterPhaseChangeModel,
)

__all__ = [
    # fvConstraints
    "FvConstraint",
    "BoundConstraint",
    "FixedValueConstraint",
    "LimitPressureConstraint",
    "LimitTemperatureConstraint",
    "create_constraint",
    # fvModels
    "FvModel",
    "SemiImplicitSource",
    "MassSource",
    "HeatSource",
    "PorosityForce",
    "CodedFvModel",
    "ActuationDiskModel",
    "HeatExchangerModel",
    "create_fv_model",
    # fvSources expansion
    "SolidificationMeltingModel",
    "RASourceModel",
    "GravitationalBodyForce",
    # enhanced_2: SemiImplicitSource 变体
    "CellSetSemiImplicitSource",
    "PatchSemiImplicitSource",
    "ExplicitSource",
    # enhanced_3: 浮力与旋转参考系力
    "BuoyancyForce",
    "BoussinesqBuoyancy",
    "SRFForce",
    # enhanced_4: 约束变体
    "MinMaxConstraint",
    "RhoLimitsConstraint",
    "VelocityLimitsConstraint",
    # enhanced_5: 辐射与多相流
    "FvDOMRadiationSource",
    "SolarLoadSource",
    "InterPhaseChangeModel",
]
