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
- :class:`MinTemperatureConstraint` — minimum temperature enforcement
- :class:`MaxTemperatureConstraint` — maximum temperature enforcement
- :class:`MassFractionLimitsConstraint` — species mass fraction bounds

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
- :class:`XiEqModel` — equilibrium flame wrinkling model
- :class:`PaSRSource` — Partially Stirred Reactor combustion source
- :class:`EDCSource` — Eddy Dissipation Concept combustion source
- :class:`MRFSource` — Multiple Reference Frame momentum source
- :class:`MRFSolidBody` — MRF solid body rotation penalty source
- :class:`InterRegionHeatTransfer` — inter-region heat transfer source
- :class:`DispersionRASource` — radiation dispersion source
- :class:`TurbulentDispersionSource` — turbulent dispersion force
- :class:`ExplicitPorositySource` — explicit porosity resistance
- :class:`RotatingDiskSource` — rotating disk momentum source
- :class:`RotatingConeSource` — rotating cone momentum source
- :class:`CodedSource` — user-coded source term

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

from pyfoam.fv.enhanced_6 import (
    XiEqModel,
    PaSRSource,
    EDCSource,
)

from pyfoam.fv.enhanced_7 import (
    MRFSource,
    MRFSolidBody,
    InterRegionHeatTransfer,
)

from pyfoam.fv.enhanced_8 import (
    MinTemperatureConstraint,
    MaxTemperatureConstraint,
    MassFractionLimitsConstraint,
)

from pyfoam.fv.enhanced_9 import (
    DispersionRASource,
    TurbulentDispersionSource,
    ExplicitPorositySource,
)

from pyfoam.fv.enhanced_10 import (
    RotatingDiskSource,
    RotatingConeSource,
    CodedSource,
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
    # enhanced_6: 燃烧与反应
    "XiEqModel",
    "PaSRSource",
    "EDCSource",
    # enhanced_7: MRF 与区域间传热
    "MRFSource",
    "MRFSolidBody",
    "InterRegionHeatTransfer",
    # enhanced_8: 温度/物种约束
    "MinTemperatureConstraint",
    "MaxTemperatureConstraint",
    "MassFractionLimitsConstraint",
    # enhanced_9: 颗粒色散与多孔介质
    "DispersionRASource",
    "TurbulentDispersionSource",
    "ExplicitPorositySource",
    # enhanced_10: 旋转体与用户编码源项
    "RotatingDiskSource",
    "RotatingConeSource",
    "CodedSource",
]
