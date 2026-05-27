"""
pyfoam.fv — Finite volume constraint framework.

Provides:

- :class:`FvConstraint` — abstract base with RTS registry and ``apply(field)`` interface
- :class:`BoundConstraint` — clamp field values to [min, max]
- :class:`FixedValueConstraint` — fix values at specified cell indices
- :class:`LimitPressureConstraint` — enforce non-negative pressure
- :class:`LimitTemperatureConstraint` — enforce physical temperature range

fvConstraints modify the solution field *after* each solver step to enforce
physical bounds, mirroring OpenFOAM's ``fvConstraints`` framework.

Usage::

    from pyfoam.fv import FvConstraint, BoundConstraint

    # Factory creation
    constraint = FvConstraint.create("bound", min=0.0, max=1.0)

    # Apply after solver step
    constraint.apply(field)
"""

from pyfoam.fv.fv_constraints import (
    FvConstraint,
    BoundConstraint,
    FixedValueConstraint,
    LimitPressureConstraint,
    LimitTemperatureConstraint,
    create_constraint,
)

__all__ = [
    "FvConstraint",
    "BoundConstraint",
    "FixedValueConstraint",
    "LimitPressureConstraint",
    "LimitTemperatureConstraint",
    "create_constraint",
]
