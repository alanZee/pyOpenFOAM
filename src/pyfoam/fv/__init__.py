"""
pyfoam.fv — Finite volume constraint & source model framework.

Provides:

**fvConstraints** (post-solution field modifications):

- :class:`FvConstraint` — abstract base with RTS registry and ``apply(field)`` interface
- :class:`BoundConstraint` — clamp field values to [min, max]
- :class:`FixedValueConstraint` — fix values at specified cell indices
- :class:`LimitPressureConstraint` — enforce non-negative pressure
- :class:`LimitTemperatureConstraint` — enforce physical temperature range

**fvModels** (pre-solution source term injection):

- :class:`FvModel` — abstract base with RTS registry and ``apply(matrix, field)`` interface
- :class:`SemiImplicitSource` — generic ``Su + Sp * phi`` volumetric source
- :class:`MassSource` — mass source / sink in continuity
- :class:`HeatSource` — volumetric heat source in energy equation
- :class:`PorosityForce` — Darcy-Forchheimer porosity resistance
- :class:`CodedFvModel` — user-defined Python function as source
- :class:`ActuationDiskModel` — actuation disk thrust source for wind turbines
- :class:`HeatExchangerModel` — temperature-dependent heat exchanger source

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
]
