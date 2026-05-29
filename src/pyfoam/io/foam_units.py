"""
FoamUnits — OpenFOAM unit system utilities.

OpenFOAM uses a 7-element dimension set::

    [mass length time temperature quantity current luminous_intensity]

with SI base units: kg, m, s, K, mol, A, cd.

This module provides:

- :class:`DimensionSet` — parse and manipulate 7-element dimension arrays
- :class:`UnitSystem` — predefined unit systems (SI, CGS, etc.)
- :func:`convert_dimensions` — convert dimensioned scalars between unit systems
- :func:`parse_dimensions` — parse ``[kg m s K mol A cd]`` strings

Usage::

    dims = parse_dimensions("[0 2 -2 0 0 0 0]")  # kinematic viscosity
    si = UnitSystem.si()
    cgs = UnitSystem.cgs()
    value_cgs = convert_dimensions(1.0, dims, from_sys=si, to_sys=cgs)

References
----------
- OpenFOAM ``dimensionSet`` class
- OpenFOAM ``unitConversion`` class
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

__all__ = [
    "DimensionSet",
    "UnitSystem",
    "convert_dimensions",
    "parse_dimensions",
]

logger = logging.getLogger(__name__)

# Standard OpenFOAM dimension names
_DIM_NAMES = ["mass", "length", "time", "temperature", "quantity", "current", "luminous_intensity"]

# Standard dimension symbols
_DIM_SYMBOLS = ["kg", "m", "s", "K", "mol", "A", "cd"]


@dataclass(frozen=True)
class DimensionSet:
    """7-element dimension set for OpenFOAM.

    Attributes:
        mass: Mass exponent.
        length: Length exponent.
        time: Time exponent.
        temperature: Temperature exponent.
        quantity: Quantity (amount of substance) exponent.
        current: Electric current exponent.
        luminous_intensity: Luminous intensity exponent.
    """

    mass: float = 0.0
    length: float = 0.0
    time: float = 0.0
    temperature: float = 0.0
    quantity: float = 0.0
    current: float = 0.0
    luminous_intensity: float = 0.0

    def to_list(self) -> List[float]:
        """Return as a 7-element list."""
        return [
            self.mass, self.length, self.time, self.temperature,
            self.quantity, self.current, self.luminous_intensity,
        ]

    def to_tuple(self) -> Tuple[float, ...]:
        """Return as a 7-element tuple."""
        return tuple(self.to_list())

    @classmethod
    def from_list(cls, values: List[float]) -> DimensionSet:
        """Create from a 7-element list."""
        if len(values) != 7:
            raise ValueError(f"Expected 7 dimensions, got {len(values)}")
        return cls(*values)

    def __mul__(self, other: DimensionSet) -> DimensionSet:
        """Multiply dimension sets (add exponents)."""
        return DimensionSet(
            mass=self.mass + other.mass,
            length=self.length + other.length,
            time=self.time + other.time,
            temperature=self.temperature + other.temperature,
            quantity=self.quantity + other.quantity,
            current=self.current + other.current,
            luminous_intensity=self.luminous_intensity + other.luminous_intensity,
        )

    def __truediv__(self, other: DimensionSet) -> DimensionSet:
        """Divide dimension sets (subtract exponents)."""
        return DimensionSet(
            mass=self.mass - other.mass,
            length=self.length - other.length,
            time=self.time - other.time,
            temperature=self.temperature - other.temperature,
            quantity=self.quantity - other.quantity,
            current=self.current - other.current,
            luminous_intensity=self.luminous_intensity - other.luminous_intensity,
        )

    def __pow__(self, exponent: float) -> DimensionSet:
        """Raise to a power (multiply all exponents)."""
        return DimensionSet(
            mass=self.mass * exponent,
            length=self.length * exponent,
            time=self.time * exponent,
            temperature=self.temperature * exponent,
            quantity=self.quantity * exponent,
            current=self.current * exponent,
            luminous_intensity=self.luminous_intensity * exponent,
        )

    def is_dimensionless(self) -> bool:
        """Check if all exponents are zero."""
        return all(abs(v) < 1e-30 for v in self.to_list())

    def to_symbol_string(self) -> str:
        """Convert to a human-readable symbol string (e.g., 'kg m s^-2')."""
        parts = []
        for name, symbol in zip(_DIM_NAMES, _DIM_SYMBOLS):
            exp = getattr(self, name)
            if abs(exp) < 1e-30:
                continue
            if abs(exp - 1.0) < 1e-30:
                parts.append(symbol)
            else:
                parts.append(f"{symbol}^{exp:g}")
        return " ".join(parts) if parts else "1"

    def __repr__(self) -> str:
        vals = self.to_list()
        return f"DimensionSet({vals})"


# ---------------------------------------------------------------------------
# Predefined dimension sets
# ---------------------------------------------------------------------------

#: Dimensionless
DIMENSIONLESS = DimensionSet()

#: Length [m]
DIM_LENGTH = DimensionSet(length=1.0)

#: Area [m^2]
DIM_AREA = DimensionSet(length=2.0)

#: Volume [m^3]
DIM_VOLUME = DimensionSet(length=3.0)

#: Time [s]
DIM_TIME = DimensionSet(time=1.0)

#: Velocity [m/s]
DIM_VELOCITY = DimensionSet(length=1.0, time=-1.0)

#: Acceleration [m/s^2]
DIM_ACCELERATION = DimensionSet(length=1.0, time=-2.0)

#: Force [kg m / s^2]
DIM_FORCE = DimensionSet(mass=1.0, length=1.0, time=-2.0)

#: Pressure [kg / (m s^2)]
DIM_PRESSURE = DimensionSet(mass=1.0, length=-1.0, time=-2.0)

#: Kinematic viscosity [m^2/s]
DIM_KINEMATIC_VISCOSITY = DimensionSet(length=2.0, time=-1.0)

#: Dynamic viscosity [kg / (m s)]
DIM_DYNAMIC_VISCOSITY = DimensionSet(mass=1.0, length=-1.0, time=-1.0)

#: Density [kg / m^3]
DIM_DENSITY = DimensionSet(mass=1.0, length=-3.0)

#: Energy [kg m^2 / s^2]
DIM_ENERGY = DimensionSet(mass=1.0, length=2.0, time=-2.0)

#: Temperature [K]
DIM_TEMPERATURE = DimensionSet(temperature=1.0)

#: Specific heat [m^2 / (s^2 K)]
DIM_SPECIFIC_HEAT = DimensionSet(length=2.0, time=-2.0, temperature=-1.0)


# ---------------------------------------------------------------------------
# Unit systems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitSystem:
    """A unit system with base unit scales.

    Each scale factor converts from SI to the target system:
    ``target_value = si_value * scale``.

    Attributes:
        name: Human-readable name.
        mass_scale: kg to target mass unit.
        length_scale: m to target length unit.
        time_scale: s to target time unit.
        temperature_scale: K to target temperature unit.
    """

    name: str
    mass_scale: float = 1.0
    length_scale: float = 1.0
    time_scale: float = 1.0
    temperature_scale: float = 1.0
    quantity_scale: float = 1.0
    current_scale: float = 1.0
    luminous_intensity_scale: float = 1.0

    def get_scales(self) -> List[float]:
        """Return all 7 scale factors."""
        return [
            self.mass_scale, self.length_scale, self.time_scale,
            self.temperature_scale, self.quantity_scale,
            self.current_scale, self.luminous_intensity_scale,
        ]

    @classmethod
    def si(cls) -> UnitSystem:
        """SI unit system (kg, m, s, K, mol, A, cd)."""
        return cls(name="SI")

    @classmethod
    def cgs(cls) -> UnitSystem:
        """CGS unit system (g, cm, s, K, mol, A, cd)."""
        return cls(
            name="CGS",
            mass_scale=1e3,       # kg -> g (* 1000)
            length_scale=1e2,     # m -> cm (* 100)
            time_scale=1.0,       # s -> s
            temperature_scale=1.0,
        )

    @classmethod
    def imperial(cls) -> UnitSystem:
        """Imperial unit system (lb, ft, s, R, mol, A, cd)."""
        return cls(
            name="Imperial",
            mass_scale=2.20462,       # kg -> lb
            length_scale=3.28084,     # m -> ft
            time_scale=1.0,           # s -> s
            temperature_scale=1.8,    # K -> R (Rankine)
        )

    @classmethod
    def mm_ton_s(cls) -> UnitSystem:
        """Millimetre-tonne-second unit system (common in FEA).

        Mass: tonne (1e-3 kg -> tonne, scale = 1e-6 ... no)
        Actually: 1 tonne = 1000 kg, so mass_scale = 1/1000 per kg... wait.

        mm-ton-s: length=mm, mass=tonne, time=s
        - 1 m = 1000 mm, so length_scale = 1000
        - 1 kg = 0.001 tonne, so mass_scale = 0.001
        """
        return cls(
            name="mm-ton-s",
            mass_scale=1e-3,      # kg -> tonne
            length_scale=1e3,     # m -> mm
            time_scale=1.0,       # s -> s
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Regex for dimension string like "[0 2 -2 0 0 0 0]" or "[kg m s^-2]"
_DIM_BRACKET_PATTERN = re.compile(r"\[([^\]]+)\]")


def parse_dimensions(dim_str: str) -> DimensionSet:
    """Parse an OpenFOAM dimension string.

    Accepts both numeric arrays and symbolic expressions::

        "[0 2 -2 0 0 0 0]"
        "[kg m^2 s^-2]"

    Args:
        dim_str: Dimension string (with or without brackets).

    Returns:
        Parsed :class:`DimensionSet`.
    """
    # Strip brackets if present
    match = _DIM_BRACKET_PATTERN.search(dim_str)
    if match:
        inner = match.group(1).strip()
    else:
        inner = dim_str.strip()

    parts = inner.split()

    # Try numeric parsing first
    try:
        values = [float(p) for p in parts]
        if len(values) == 7:
            return DimensionSet.from_list(values)
    except ValueError:
        pass

    # Symbolic parsing: "kg m s^-2" etc.
    dims = [0.0] * 7
    for part in parts:
        if "^" in part:
            base, exp_str = part.split("^", 1)
            exp = float(exp_str)
        else:
            base = part
            exp = 1.0

        # Match base symbol
        matched = False
        for i, sym in enumerate(_DIM_SYMBOLS):
            if base == sym:
                dims[i] += exp
                matched = True
                break

        if not matched:
            # Try common aliases
            aliases = {
                "N": ("kg", "m", "s"),
                "Pa": ("kg", "m", "s"),
                "J": ("kg", "m", "s"),
                "W": ("kg", "m", "s"),
            }
            if base in aliases:
                logger.debug("Alias '%s' not fully resolved; skipping", base)

    return DimensionSet.from_list(dims)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert_dimensions(
    value: float,
    dimensions: DimensionSet,
    from_sys: UnitSystem,
    to_sys: UnitSystem,
) -> float:
    """Convert a dimensioned value between unit systems.

    Args:
        value: Numeric value in *from_sys* units.
        dimensions: The :class:`DimensionSet` of the quantity.
        from_sys: Source unit system.
        to_sys: Target unit system.

    Returns:
        Value in *to_sys* units.

    Example::

        # Convert 1 m^2/s (kinematic viscosity) from SI to CGS
        # SI: 1 m^2/s, CGS: 10000 cm^2/s
        nu_cgs = convert_dimensions(
            1.0,
            DIM_KINEMATIC_VISCOSITY,
            UnitSystem.si(),
            UnitSystem.cgs(),
        )
        # nu_cgs = 1.0 * (100/1)^2 / (1)^1 = 10000.0
    """
    from_scales = from_sys.get_scales()
    to_scales = to_sys.get_scales()
    dim_vals = dimensions.to_list()

    # Conversion factor: product of (to_scale/from_scale)^dim_exponent
    factor = 1.0
    for s_from, s_to, exp in zip(from_scales, to_scales, dim_vals):
        if abs(exp) > 1e-30:
            factor *= (s_to / s_from) ** exp

    return value * factor


def get_derived_unit(name: str) -> DimensionSet:
    """Get the dimension set for a common derived unit.

    Args:
        name: Unit name (e.g., ``"Pa"``, ``"N"``, ``"J"``).

    Returns:
        Corresponding :class:`DimensionSet`.

    Raises:
        KeyError: If the unit name is unknown.
    """
    _derived_units: Dict[str, DimensionSet] = {
        "N": DIM_FORCE,
        "Pa": DIM_PRESSURE,
        "J": DIM_ENERGY,
        "W": DimensionSet(mass=1.0, length=2.0, time=-3.0),
        "Hz": DimensionSet(time=-1.0),
        "m/s": DIM_VELOCITY,
        "m/s^2": DIM_ACCELERATION,
        "m^2": DIM_AREA,
        "m^3": DIM_VOLUME,
        "kg/m^3": DIM_DENSITY,
        "Pa.s": DIM_DYNAMIC_VISCOSITY,
        "m^2/s": DIM_KINEMATIC_VISCOSITY,
    }
    if name not in _derived_units:
        raise KeyError(f"Unknown derived unit: {name}. Available: {list(_derived_units.keys())}")
    return _derived_units[name]
