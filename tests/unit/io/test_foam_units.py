"""Tests for FoamUnits — unit system utilities and dimension handling."""

import pytest

from pyfoam.io.foam_units import (
    DimensionSet,
    UnitSystem,
    convert_dimensions,
    parse_dimensions,
    get_derived_unit,
    DIMENSIONLESS,
    DIM_LENGTH,
    DIM_AREA,
    DIM_VOLUME,
    DIM_TIME,
    DIM_VELOCITY,
    DIM_ACCELERATION,
    DIM_FORCE,
    DIM_PRESSURE,
    DIM_KINEMATIC_VISCOSITY,
    DIM_DYNAMIC_VISCOSITY,
    DIM_DENSITY,
    DIM_ENERGY,
    DIM_TEMPERATURE,
    DIM_SPECIFIC_HEAT,
)


# ---------------------------------------------------------------------------
# DimensionSet tests
# ---------------------------------------------------------------------------


class TestDimensionSet:
    """Test DimensionSet dataclass."""

    def test_default_is_dimensionless(self):
        """Default DimensionSet is dimensionless."""
        d = DimensionSet()
        assert d.is_dimensionless()
        assert d.mass == 0.0
        assert d.length == 0.0
        assert d.time == 0.0

    def test_custom_dimensions(self):
        """Custom dimension set."""
        d = DimensionSet(mass=1.0, length=-1.0, time=-2.0)
        assert d.mass == 1.0
        assert d.length == -1.0
        assert d.time == -2.0
        assert not d.is_dimensionless()

    def test_to_list(self):
        """to_list returns 7 elements."""
        d = DimensionSet(mass=1.0, length=2.0)
        lst = d.to_list()
        assert len(lst) == 7
        assert lst == [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_to_tuple(self):
        """to_tuple returns 7 elements."""
        d = DimensionSet(time=1.0)
        tup = d.to_tuple()
        assert len(tup) == 7
        assert tup[2] == 1.0

    def test_from_list(self):
        """Create from list."""
        d = DimensionSet.from_list([1, 2, -1, 0, 0, 0, 0])
        assert d.mass == 1.0
        assert d.length == 2.0
        assert d.time == -1.0

    def test_from_list_wrong_length_raises(self):
        """ValueError for wrong-length list."""
        with pytest.raises(ValueError, match="Expected 7"):
            DimensionSet.from_list([1, 2, 3])

    def test_multiply(self):
        """Multiply dimension sets."""
        d1 = DIM_LENGTH       # [0 1 0 0 0 0 0]
        d2 = DIM_TIME         # [0 0 1 0 0 0 0]
        d3 = d1 * d2          # [0 1 1 0 0 0 0] = length * time
        assert d3.length == 1.0
        assert d3.time == 1.0
        assert d3.mass == 0.0

    def test_divide(self):
        """Divide dimension sets."""
        d = DIM_LENGTH / DIM_TIME  # [0 1 -1 ...] = velocity
        assert d.length == 1.0
        assert d.time == -1.0
        assert d == DIM_VELOCITY

    def test_power(self):
        """Raise to power."""
        d = DIM_LENGTH ** 2
        assert d.length == 2.0
        assert d == DIM_AREA

    def test_to_symbol_string(self):
        """Human-readable symbol string."""
        d = DIM_PRESSURE
        s = d.to_symbol_string()
        assert "kg" in s
        assert "m" in s
        assert "s" in s

    def test_dimensionless_symbol_string(self):
        """Dimensionless quantity returns '1'."""
        assert DIMENSIONLESS.to_symbol_string() == "1"

    def test_repr(self):
        """repr includes values."""
        r = repr(DIM_LENGTH)
        assert "DimensionSet" in r


# ---------------------------------------------------------------------------
# Predefined dimension tests
# ---------------------------------------------------------------------------


class TestPredefinedDimensions:
    """Test predefined dimension constants."""

    def test_velocity_dims(self):
        d = DIM_VELOCITY
        assert d.length == 1.0
        assert d.time == -1.0
        assert d.mass == 0.0

    def test_pressure_dims(self):
        d = DIM_PRESSURE
        assert d.mass == 1.0
        assert d.length == -1.0
        assert d.time == -2.0

    def test_kinematic_viscosity_dims(self):
        d = DIM_KINEMATIC_VISCOSITY
        assert d.length == 2.0
        assert d.time == -1.0

    def test_energy_dims(self):
        d = DIM_ENERGY
        assert d.mass == 1.0
        assert d.length == 2.0
        assert d.time == -2.0

    def test_temperature_is_pure(self):
        d = DIM_TEMPERATURE
        assert d.temperature == 1.0
        assert d.mass == 0.0


# ---------------------------------------------------------------------------
# UnitSystem tests
# ---------------------------------------------------------------------------


class TestUnitSystem:
    """Test unit system definitions."""

    def test_si_defaults(self):
        """SI has all scales = 1.0."""
        si = UnitSystem.si()
        assert si.name == "SI"
        assert all(s == 1.0 for s in si.get_scales())

    def test_cgs_scales(self):
        """CGS has correct scales."""
        cgs = UnitSystem.cgs()
        assert cgs.name == "CGS"
        assert cgs.mass_scale == 1e3     # kg -> g
        assert cgs.length_scale == 1e2   # m -> cm

    def test_mm_ton_s_scales(self):
        """mm-ton-s has correct scales."""
        mts = UnitSystem.mm_ton_s()
        assert mts.name == "mm-ton-s"
        assert mts.mass_scale == 1e-3    # kg -> tonne
        assert mts.length_scale == 1e3   # m -> mm

    def test_get_scales_length(self):
        """get_scales returns 7 elements."""
        si = UnitSystem.si()
        assert len(si.get_scales()) == 7

    def test_imperial_scales(self):
        """Imperial has reasonable scales."""
        imp = UnitSystem.imperial()
        assert imp.name == "Imperial"
        assert imp.length_scale > 3.0    # m -> ft
        assert imp.mass_scale > 2.0      # kg -> lb


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


class TestParseDimensions:
    """Test dimension string parsing."""

    def test_parse_numeric(self):
        """Parse numeric dimension string."""
        d = parse_dimensions("[0 2 -2 0 0 0 0]")
        assert d.length == 2.0
        assert d.time == -2.0
        assert d.mass == 0.0

    def test_parse_without_brackets(self):
        """Parse without brackets."""
        d = parse_dimensions("0 1 0 0 0 0 0")
        assert d.length == 1.0

    def test_parse_kinematic_viscosity(self):
        """Parse kinematic viscosity dimensions."""
        d = parse_dimensions("[0 2 -1 0 0 0 0]")
        assert d == DIM_KINEMATIC_VISCOSITY

    def test_parse_pressure(self):
        """Parse pressure dimensions."""
        d = parse_dimensions("[1 -1 -2 0 0 0 0]")
        assert d == DIM_PRESSURE

    def test_parse_symbolic(self):
        """Parse symbolic dimension string."""
        d = parse_dimensions("[kg m s^-2]")
        assert d.mass == 1.0
        assert d.length == 1.0
        assert d.time == -2.0


# ---------------------------------------------------------------------------
# Conversion tests
# ---------------------------------------------------------------------------


class TestConvertDimensions:
    """Test unit conversion."""

    def test_si_to_si_identity(self):
        """SI to SI is identity."""
        val = convert_dimensions(1.0, DIM_LENGTH, UnitSystem.si(), UnitSystem.si())
        assert val == 1.0

    def test_si_to_cgs_length(self):
        """1 m -> 100 cm."""
        val = convert_dimensions(1.0, DIM_LENGTH, UnitSystem.si(), UnitSystem.cgs())
        assert val == pytest.approx(100.0)

    def test_si_to_cgs_area(self):
        """1 m^2 -> 10000 cm^2."""
        val = convert_dimensions(1.0, DIM_AREA, UnitSystem.si(), UnitSystem.cgs())
        assert val == pytest.approx(10000.0)

    def test_si_to_cgs_velocity(self):
        """1 m/s -> 100 cm/s."""
        val = convert_dimensions(1.0, DIM_VELOCITY, UnitSystem.si(), UnitSystem.cgs())
        assert val == pytest.approx(100.0)

    def test_si_to_cgs_pressure(self):
        """1 Pa -> 10 Ba (dyne/cm^2)."""
        val = convert_dimensions(1.0, DIM_PRESSURE, UnitSystem.si(), UnitSystem.cgs())
        assert val == pytest.approx(10.0)

    def test_si_to_cgs_kinematic_viscosity(self):
        """1 m^2/s -> 10000 cm^2/s."""
        val = convert_dimensions(
            1.0, DIM_KINEMATIC_VISCOSITY, UnitSystem.si(), UnitSystem.cgs()
        )
        assert val == pytest.approx(10000.0)

    def test_cgs_to_si_length(self):
        """100 cm -> 1 m."""
        val = convert_dimensions(100.0, DIM_LENGTH, UnitSystem.cgs(), UnitSystem.si())
        assert val == pytest.approx(1.0)

    def test_roundtrip(self):
        """SI -> CGS -> SI roundtrip."""
        original = 42.0
        cgs = convert_dimensions(original, DIM_PRESSURE, UnitSystem.si(), UnitSystem.cgs())
        back = convert_dimensions(cgs, DIM_PRESSURE, UnitSystem.cgs(), UnitSystem.si())
        assert back == pytest.approx(original)

    def test_dimensionless_conversion(self):
        """Dimensionless conversion is identity."""
        val = convert_dimensions(5.0, DIMENSIONLESS, UnitSystem.si(), UnitSystem.cgs())
        assert val == 5.0

    def test_mm_ton_s_conversion(self):
        """SI to mm-ton-s conversion."""
        # 1 m = 1000 mm
        val = convert_dimensions(1.0, DIM_LENGTH, UnitSystem.si(), UnitSystem.mm_ton_s())
        assert val == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Derived unit tests
# ---------------------------------------------------------------------------


class TestDerivedUnits:
    """Test derived unit lookup."""

    def test_newton(self):
        d = get_derived_unit("N")
        assert d == DIM_FORCE

    def test_pascal(self):
        d = get_derived_unit("Pa")
        assert d == DIM_PRESSURE

    def test_joule(self):
        d = get_derived_unit("J")
        assert d == DIM_ENERGY

    def test_hertz(self):
        d = get_derived_unit("Hz")
        assert d.time == -1.0

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown derived unit"):
            get_derived_unit("unknown_unit")
