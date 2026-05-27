"""
fvConstraints — post-solution constraint framework.

Provides an RTS (Run-Time Selection) registry of constraints that modify
field values after each solver iteration to enforce physical bounds.

In OpenFOAM, ``fvConstraints`` are applied in the solution loop after the
linear solver returns, ensuring physical realizability of fields such as
pressure (non-negative) and temperature (within material limits).

Constraints:

- **bound** — clamp field to ``[min, max]``
- **fixedValue** — fix values at specified cell indices
- **limitPressure** — enforce ``p >= min`` (default 0)
- **limitTemperature** — enforce ``T_min <= T <= T_max``

Usage::

    from pyfoam.fv.fv_constraints import FvConstraint, BoundConstraint

    # Decorator-based registration
    @FvConstraint.register("myConstraint")
    class MyConstraint(FvConstraint):
        ...

    # Factory creation
    c = FvConstraint.create("bound", min=0.0, max=1.0)
    c.apply(field)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Type

import torch

__all__ = [
    "FvConstraint",
    "BoundConstraint",
    "FixedValueConstraint",
    "LimitPressureConstraint",
    "LimitTemperatureConstraint",
    "create_constraint",
]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class FvConstraint(ABC):
    """Abstract base for all fvConstraints.

    Subclasses must implement :meth:`apply`.

    RTS (Run-Time Selection) registry allows string-based lookup::

        @FvConstraint.register("bound")
        class BoundConstraint(FvConstraint):
            ...

        c = FvConstraint.create("bound", min=0.0, max=1.0)
    """

    _registry: ClassVar[dict[str, Type[FvConstraint]]] = {}

    def __init__(self, **kwargs: Any) -> None:
        self._coeffs: dict[str, Any] = kwargs

    # ------------------------------------------------------------------
    # RTS registry
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a constraint class under *name*.

        Usage::

            @FvConstraint.register("bound")
            class BoundConstraint(FvConstraint):
                ...
        """

        def decorator(constraint_cls: Type[FvConstraint]) -> Type[FvConstraint]:
            if name in cls._registry:
                raise ValueError(
                    f"fvConstraint '{name}' is already registered "
                    f"to {cls._registry[name].__name__}"
                )
            cls._registry[name] = constraint_cls
            return constraint_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> FvConstraint:
        """Factory: create a constraint instance by registered *name*.

        Args:
            name: Registered constraint type name (e.g. ``"bound"``).
            **kwargs: Constraint parameters (e.g. ``min``, ``max``, ``cells``).

        Returns:
            Instantiated constraint.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise KeyError(
                f"Unknown fvConstraint type '{name}'. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return sorted list of registered constraint type names."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def type_name(self) -> str:
        """Return the registered type name for this constraint class."""
        for name, constraint_cls in self._registry.items():
            if isinstance(self, constraint_cls):
                return name
        return self.__class__.__name__

    @property
    def coeffs(self) -> dict[str, Any]:
        """Return the constraint coefficient dictionary."""
        return self._coeffs

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """Apply the constraint to *field*, modifying values in-place.

        Args:
            field: Solution field tensor (1-D for scalar, 2-D for vector).

        Returns:
            The (possibly modified) field tensor for chaining convenience.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._coeffs})"


# ---------------------------------------------------------------------------
# BoundConstraint
# ---------------------------------------------------------------------------


@FvConstraint.register("bound")
class BoundConstraint(FvConstraint):
    """Clamp field values to ``[min, max]``.

    Corresponds to OpenFOAM's ``fvConstraints.bound``.

    Parameters
    ----------
    min : float | None
        Lower bound.  ``None`` means no lower bound.
    max : float | None
        Upper bound.  ``None`` means no upper bound.

    Examples::

        c = BoundConstraint(min=0.0, max=1.0)
        c.apply(field)
    """

    def __init__(self, *, min: float | None = None, max: float | None = None, **kwargs: Any) -> None:
        super().__init__(min=min, max=max, **kwargs)
        self._min = min
        self._max = max

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """Clamp field to ``[min, max]``."""
        if self._min is not None:
            field.clamp_(min=self._min)
        if self._max is not None:
            field.clamp_(max=self._max)
        return field

    def __repr__(self) -> str:
        return f"BoundConstraint(min={self._min}, max={self._max})"


# ---------------------------------------------------------------------------
# FixedValueConstraint
# ---------------------------------------------------------------------------


@FvConstraint.register("fixedValue")
class FixedValueConstraint(FvConstraint):
    """Fix field values at specified cell indices.

    Parameters
    ----------
    cells : list[int] | torch.Tensor
        Cell indices at which to fix the value.
    value : float
        The value to set at those cells.

    Examples::

        c = FixedValueConstraint(cells=[0, 1, 2], value=101325.0)
        c.apply(p_field)
    """

    def __init__(
        self,
        *,
        cells: list[int] | torch.Tensor,
        value: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(cells=cells, value=value, **kwargs)
        if isinstance(cells, list):
            self._cells = torch.tensor(cells, dtype=torch.long)
        else:
            self._cells = cells.to(dtype=torch.long)
        self._value = value

    @property
    def cells(self) -> torch.Tensor:
        """Cell indices to fix."""
        return self._cells

    @property
    def value(self) -> float:
        """Fixed value."""
        return self._value

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """Set field values at specified cells."""
        field[self._cells] = self._value
        return field

    def __repr__(self) -> str:
        return (
            f"FixedValueConstraint(cells={self._cells.tolist()}, "
            f"value={self._value})"
        )


# ---------------------------------------------------------------------------
# LimitPressureConstraint
# ---------------------------------------------------------------------------


@FvConstraint.register("limitPressure")
class LimitPressureConstraint(FvConstraint):
    """Enforce pressure >= min (default 0).

    Corresponds to OpenFOAM's ``limitPressure`` fvConstraint.
    Clamps pressure to be non-negative (or above a specified minimum).

    Parameters
    ----------
    min : float
        Minimum allowable pressure.  Default ``0.0``.

    Examples::

        c = LimitPressureConstraint(min=0.0)
        c.apply(p_field)
    """

    def __init__(self, *, min: float = 0.0, **kwargs: Any) -> None:
        super().__init__(min=min, **kwargs)
        self._min = min

    @property
    def min(self) -> float:
        """Minimum pressure."""
        return self._min

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """Clamp pressure to ``>= min``."""
        field.clamp_(min=self._min)
        return field

    def __repr__(self) -> str:
        return f"LimitPressureConstraint(min={self._min})"


# ---------------------------------------------------------------------------
# LimitTemperatureConstraint
# ---------------------------------------------------------------------------


@FvConstraint.register("limitTemperature")
class LimitTemperatureConstraint(FvConstraint):
    """Enforce temperature within physical range ``[min, max]``.

    Corresponds to OpenFOAM's ``limitTemperature`` fvConstraint.

    Parameters
    ----------
    min : float
        Minimum temperature.  Default ``1.0`` (to avoid T=0 issues).
    max : float
        Maximum temperature.  Default ``5000.0``.

    Examples::

        c = LimitTemperatureConstraint(min=200.0, max=5000.0)
        c.apply(T_field)
    """

    def __init__(self, *, min: float = 1.0, max: float = 5000.0, **kwargs: Any) -> None:
        super().__init__(min=min, max=max, **kwargs)
        self._min = min
        self._max = max

    @property
    def min(self) -> float:
        """Minimum temperature."""
        return self._min

    @property
    def max(self) -> float:
        """Maximum temperature."""
        return self._max

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """Clamp temperature to ``[min, max]``."""
        field.clamp_(min=self._min, max=self._max)
        return field

    def __repr__(self) -> str:
        return f"LimitTemperatureConstraint(min={self._min}, max={self._max})"


# ---------------------------------------------------------------------------
# Factory convenience function
# ---------------------------------------------------------------------------


def create_constraint(name: str, **kwargs: Any) -> FvConstraint:
    """Create an fvConstraint by registered name.

    Convenience wrapper around :meth:`FvConstraint.create`.

    Args:
        name: Constraint type name (case-insensitive).
        **kwargs: Constraint parameters.

    Returns:
        Instantiated constraint.
    """
    return FvConstraint.create(name, **kwargs)
