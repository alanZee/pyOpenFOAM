"""Tests for FieldMinMax — field min/max analysis."""

import pytest
import torch

from pyfoam.postprocessing.field_min_max import FieldMinMax, MinMaxResult


# ---------------------------------------------------------------------------
# MinMaxResult tests
# ---------------------------------------------------------------------------


class TestMinMaxResult:
    """Test MinMaxResult dataclass."""

    def test_defaults(self):
        r = MinMaxResult()
        assert r.time == 0.0
        assert r.field_name == ""
        assert r.min_location == -1
        assert r.max_location == -1

    def test_custom_values(self):
        r = MinMaxResult(
            time=1.0, field_name="p",
            min_value=100.0, max_value=200.0,
            min_location=5, max_location=10,
        )
        assert r.time == 1.0
        assert r.min_value == 100.0
        assert r.max_location == 10


# ---------------------------------------------------------------------------
# FieldMinMax function object tests
# ---------------------------------------------------------------------------


class TestFieldMinMax:
    """Test FieldMinMax function object."""

    def test_creation(self):
        """Create FieldMinMax with config."""
        fmm = FieldMinMax("testMinMax", {"field": "p"})
        assert fmm.name == "testMinMax"
        assert fmm._field_name == "p"

    def test_default_log(self):
        """Default log is True."""
        fmm = FieldMinMax()
        assert fmm._do_log is True

    def test_initialise(self, fv_mesh, sample_fields):
        """Initialise with mesh and fields."""
        fmm = FieldMinMax("test", {"field": "p"})
        fmm.initialise(fv_mesh, sample_fields)
        assert fmm._mesh is fv_mesh

    def test_execute_scalar_field(self, fv_mesh, sample_fields):
        """Execute on scalar field."""
        fmm = FieldMinMax("test", {"field": "p"})
        fmm.initialise(fv_mesh, sample_fields)
        fmm.execute(1.0)

        assert len(fmm.results) == 1
        r = fmm.results[0]
        assert r.field_name == "p"
        assert r.min_value < r.max_value
        assert r.time == 1.0

    def test_execute_vector_field(self, fv_mesh, sample_fields):
        """Execute on vector field (uses magnitude)."""
        fmm = FieldMinMax("test", {"field": "U"})
        fmm.initialise(fv_mesh, sample_fields)
        fmm.execute(1.0)

        assert len(fmm.results) == 1
        r = fmm.results[0]
        assert r.min_value >= 0.0  # magnitude is non-negative

    def test_execute_multiple_steps(self, fv_mesh, sample_fields):
        """Execute at multiple time steps."""
        fmm = FieldMinMax("test", {"field": "p"})
        fmm.initialise(fv_mesh, sample_fields)

        fmm.execute(0.0)
        fmm.execute(1.0)
        fmm.execute(2.0)

        assert len(fmm.results) == 3
        assert fmm.results[0].time == 0.0
        assert fmm.results[2].time == 2.0

    def test_missing_field_no_error(self, fv_mesh, sample_fields):
        """Missing field logs warning, no crash."""
        fmm = FieldMinMax("test", {"field": "nonexistent"})
        fmm.initialise(fv_mesh, sample_fields)
        fmm.execute(1.0)

        assert len(fmm.results) == 0

    def test_disabled_no_execute(self, fv_mesh, sample_fields):
        """Disabled function object does nothing."""
        fmm = FieldMinMax("test", {"field": "p", "enabled": False})
        fmm.initialise(fv_mesh, sample_fields)
        fmm.execute(1.0)

        assert len(fmm.results) == 0

    def test_min_max_location_indices(self, fv_mesh, sample_fields):
        """Min/max locations are valid cell indices."""
        fmm = FieldMinMax("test", {"field": "p"})
        fmm.initialise(fv_mesh, sample_fields)
        fmm.execute(1.0)

        r = fmm.results[0]
        assert 0 <= r.min_location < 2  # 2 cells
        assert 0 <= r.max_location < 2

    def test_write(self, fv_mesh, sample_fields, tmp_path):
        """Write results to file."""
        fmm = FieldMinMax("test", {"field": "p"})
        fmm.set_output_path(tmp_path)
        fmm.initialise(fv_mesh, sample_fields)
        fmm.execute(1.0)
        fmm.write()

        outfile = tmp_path / "fieldMinMax.dat"
        assert outfile.exists()
        content = outfile.read_text()
        assert "Time" in content

    def test_registry(self):
        """FieldMinMax is registered."""
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        assert "fieldMinMax" in FunctionObjectRegistry.list_registered()
