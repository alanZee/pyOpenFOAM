"""
DimensionSet — physical dimensions for CFD fields.

OpenFOAM uses a 7-element dimension system::

    [mass length time temperature quantity current luminous_intensity]
    [kg    m      s    K           mol      A      cd]

Each field carries a :class:`DimensionSet` that is checked during arithmetic
to prevent physically inconsistent operations (e.g. adding velocity to pressure).

Examples::

    dim_length = DimensionSet(length=1)            # [0 1 0 0 0 0 0]
    dim_velocity = DimensionSet(length=1, time=-1) # [0 1 -1 0 0 0 0]
    dim_pressure = DimensionSet(mass=1, length=-1, time=-2)  # [1 -1 -2 0 0 0 0]
    dimless = DimensionSet()                        # [0 0 0 0 0 0 0]
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["DimensionSet"]

# Index constants for the 7 base units
_MASS = 0
_LENGTH = 1
_TIME = 2
_TEMPERATURE = 3
_QUANTITY = 4
_CURRENT = 5
_LUMINOUS_INTENSITY = 6

_NAMES = [
    "mass",
    "length",
    "time",
    "temperature",
    "quantity",
    "current",
    "luminousIntensity",
]

_SYMBOLS = ["kg", "m", "s", "K", "mol", "A", "cd"]


class DimensionSet:
    """Immutable 7-element physical dimension set.

    Parameters are keyword-only exponents for each base SI unit.
    All default to 0 (dimensionless).

    Args:
        mass: Exponent for kg.
        length: Exponent for m.
        time: Exponent for s.
        temperature: Exponent for K.
        quantity: Exponent for mol.
        current: Exponent for A.
        luminous_intensity: Exponent for cd.

    Attributes:
        exponents: Tuple of 7 float exponents.
    """

    __slots__ = ("_exponents",)

    def __init__(
        self,
        mass: float = 0,
        length: float = 0,
        time: float = 0,
        temperature: float = 0,
        quantity: float = 0,
        current: float = 0,
        luminous_intensity: float = 0,
    ) -> None:
        self._exponents: tuple[float, ...] = (
            float(mass),
            float(length),
            float(time),
            float(temperature),
            float(quantity),
            float(current),
            float(luminous_intensity),
        )

    # ------------------------------------------------------------------
    # Class-level constants for common dimensions
    # ------------------------------------------------------------------

    @classmethod
    def dimless(cls) -> "DimensionSet":
        """Return a dimensionless set (all zeros)."""
        return cls()

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "DimensionSet":
        """Create from a 7-element list ``[mass, length, time, ...]``.

        Raises:
            ValueError: If *values* does not have exactly 7 elements.
        """
        if len(values) != 7:
            raise ValueError(
                f"DimensionSet requires exactly 7 exponents, got {len(values)}"
            )
        return cls(*values)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def exponents(self) -> tuple[float, ...]:
        """The 7-element exponent tuple (immutable)."""
        return self._exponents

    @property
    def mass(self) -> float:
        return self._exponents[_MASS]

    @property
    def length(self) -> float:
        return self._exponents[_LENGTH]

    @property
    def time(self) -> float:
        return self._exponents[_TIME]

    @property
    def temperature(self) -> float:
        return self._exponents[_TEMPERATURE]

    @property
    def quantity(self) -> float:
        return self._exponents[_QUANTITY]

    @property
    def current(self) -> float:
        return self._exponents[_CURRENT]

    @property
    def luminous_intensity(self) -> float:
        return self._exponents[_LUMINOUS_INTENSITY]

    @property
    def is_dimless(self) -> bool:
        """Return ``True`` if all exponents are zero."""
        return all(e == 0.0 for e in self._exponents)

    # ------------------------------------------------------------------
    # Arithmetic — returns new DimensionSet
    # ------------------------------------------------------------------

    def __add__(self, other: "DimensionSet") -> "DimensionSet":
        """Dimensions must be identical for addition (same physical quantity)."""
        if not isinstance(other, DimensionSet):
            return NotImplemented
        if self._exponents != other._exponents:
            raise DimensionError(
                f"Cannot add fields with different dimensions: "
                f"{self} + {other}"
            )
        return DimensionSet(*self._exponents)

    def __sub__(self, other: "DimensionSet") -> "DimensionSet":
        """Dimensions must be identical for subtraction."""
        if not isinstance(other, DimensionSet):
            return NotImplemented
        if self._exponents != other._exponents:
            raise DimensionError(
                f"Cannot subtract fields with different dimensions: "
                f"{self} - {other}"
            )
        return DimensionSet(*self._exponents)

    def __mul__(self, other: "DimensionSet") -> "DimensionSet":
        """Multiplication adds exponents."""
        if not isinstance(other, DimensionSet):
            return NotImplemented
        return DimensionSet(
            *(a + b for a, b in zip(self._exponents, other._exponents))
        )

    def __truediv__(self, other: "DimensionSet") -> "DimensionSet":
        """Division subtracts exponents."""
        if not isinstance(other, DimensionSet):
            return NotImplemented
        return DimensionSet(
            *(a - b for a, b in zip(self._exponents, other._exponents))
        )

    def __neg__(self) -> "DimensionSet":
        """Negate all exponents (for reciprocal dimensions)."""
        return DimensionSet(*(-e for e in self._exponents))

    def __pow__(self, power: float) -> "DimensionSet":
        """Raise to a scalar power (multiply all exponents)."""
        return DimensionSet(*(e * power for e in self._exponents))

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DimensionSet):
            return NotImplemented
        return self._exponents == other._exponents

    def __hash__(self) -> int:
        return hash(self._exponents)

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DimensionSet({', '.join(f'{n}={e}' for n, e in zip(_NAMES, self._exponents))})"

    def __str__(self) -> str:
        """OpenFOAM-style: ``[kg m s K mol A cd]``."""
        parts = []
        for sym, exp in zip(_SYMBOLS, self._exponents):
            if exp == int(exp):
                parts.append(str(int(exp)))
            else:
                parts.append(str(exp))
        return f"[{' '.join(parts)}]"

    def to_list(self) -> list[float]:
        """Return exponents as a plain list."""
        return list(self._exponents)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DimensionError(Exception):
    """Raised when an operation violates dimensional consistency."""
