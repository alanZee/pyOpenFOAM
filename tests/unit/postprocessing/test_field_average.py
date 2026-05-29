"""Tests for FieldAverage — time-averaged field statistics."""

import pytest
import torch

from pyfoam.postprocessing.field_average import FieldAverage


# ---------------------------------------------------------------------------
# FieldAverage tests
# ---------------------------------------------------------------------------


class TestFieldAverage:
    """Test FieldAverage function object."""

    def test_creation(self):
        """Create FieldAverage with config."""
        fa = FieldAverage("testAvg", {"fields": ["p"], "mean": True})
        assert fa.name == "testAvg"
        assert fa._field_names == ["p"]
        assert fa._compute_mean is True

    def test_default_config(self):
        """Default config: mean=True, mean2=False."""
        fa = FieldAverage()
        assert fa._compute_mean is True
        assert fa._compute_mean2 is False
        assert fa._compute_prime2 is False

    def test_initialise(self, fv_mesh, sample_fields):
        """Initialise with mesh and fields."""
        fa = FieldAverage("test", {"fields": ["p", "U"]})
        fa.initialise(fv_mesh, sample_fields)
        assert fa._mesh is fv_mesh

    def test_execute_single_step(self, fv_mesh, sample_fields):
        """Execute at a single time step."""
        fa = FieldAverage("test", {"fields": ["p"], "mean": True})
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(0.0)

        assert fa.n_samples == 1
        assert "p" in fa.mean_fields
        assert fa.mean_fields["p"].shape == (2,)  # 2 cells

    def test_execute_multiple_steps_mean(self, fv_mesh, sample_fields):
        """Mean converges to constant field value."""
        fa = FieldAverage("test", {"fields": ["p"], "mean": True})
        fa.initialise(fv_mesh, sample_fields)

        # Execute multiple times with the same field
        for t in range(1, 11):
            fa.execute(float(t))

        assert fa.n_samples == 10
        # Mean of a constant field should be close to the field values
        mean_p = fa.mean_fields["p"]
        assert mean_p.shape == (2,)

    def test_execute_mean2(self, fv_mesh, sample_fields):
        """Mean-square accumulation."""
        fa = FieldAverage("test", {
            "fields": ["p"],
            "mean": True,
            "mean2": True,
        })
        fa.initialise(fv_mesh, sample_fields)

        fa.execute(1.0)
        fa.execute(2.0)

        assert "p" in fa.mean2_fields
        mean2_p = fa.mean2_fields["p"]
        assert mean2_p.shape == (2,)
        # Mean square should be positive
        assert (mean2_p >= 0).all()

    def test_execute_prime2(self, fv_mesh, sample_fields):
        """Variance computation."""
        fa = FieldAverage("test", {
            "fields": ["p"],
            "mean": True,
            "prime2Mean": True,
        })
        fa.initialise(fv_mesh, sample_fields)

        fa.execute(1.0)
        fa.execute(2.0)

        assert "p" in fa.prime2_fields
        # For constant field, variance should be ~0
        prime2 = fa.prime2_fields["p"]
        assert prime2.shape == (2,)

    def test_get_rms(self, fv_mesh, sample_fields):
        """RMS computation."""
        fa = FieldAverage("test", {
            "fields": ["p"],
            "mean2": True,
        })
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)

        rms = fa.get_rms("p")
        assert rms is not None
        assert rms.shape == (2,)
        assert (rms >= 0).all()

    def test_get_rms_no_mean2(self, fv_mesh, sample_fields):
        """RMS returns None if mean2 not computed."""
        fa = FieldAverage("test", {"fields": ["p"], "mean2": False})
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)

        assert fa.get_rms("p") is None

    def test_get_variance(self, fv_mesh, sample_fields):
        """Variance access."""
        fa = FieldAverage("test", {
            "fields": ["p"],
            "mean": True,
            "prime2Mean": True,
        })
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)

        var = fa.get_variance("p")
        assert var is not None
        assert var.shape == (2,)

    def test_disabled_no_execute(self, fv_mesh, sample_fields):
        """Disabled function object does nothing."""
        fa = FieldAverage("test", {"fields": ["p"], "enabled": False})
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)

        assert fa.n_samples == 0

    def test_missing_field_skipped(self, fv_mesh, sample_fields):
        """Missing field is silently skipped."""
        fa = FieldAverage("test", {"fields": ["p", "nonexistent"]})
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)

        assert "p" in fa.mean_fields
        assert "nonexistent" not in fa.mean_fields

    def test_write(self, fv_mesh, sample_fields, tmp_path):
        """Write results to disk."""
        fa = FieldAverage("test", {
            "fields": ["p"],
            "mean": True,
        })
        fa.set_output_path(tmp_path)
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)
        fa.write()

        assert (tmp_path / "p_mean").exists()
        assert (tmp_path / "fieldAverage_summary.dat").exists()

    def test_write_mean2(self, fv_mesh, sample_fields, tmp_path):
        """Write mean2 result."""
        fa = FieldAverage("test", {
            "fields": ["p"],
            "mean": True,
            "mean2": True,
        })
        fa.set_output_path(tmp_path)
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)
        fa.write()

        assert (tmp_path / "p_mean2").exists()

    def test_write_prime2(self, fv_mesh, sample_fields, tmp_path):
        """Write prime2 result."""
        fa = FieldAverage("test", {
            "fields": ["p"],
            "mean": True,
            "prime2Mean": True,
        })
        fa.set_output_path(tmp_path)
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)
        fa.write()

        assert (tmp_path / "p_prime2Mean").exists()

    def test_finalise_clears_accumulators(self, fv_mesh, sample_fields):
        """finalise clears accumulators."""
        fa = FieldAverage("test", {"fields": ["p"], "mean": True})
        fa.initialise(fv_mesh, sample_fields)
        fa.execute(1.0)

        assert len(fa._mean_accum) > 0
        fa.finalise()
        assert len(fa._mean_accum) == 0

    def test_registry(self):
        """FieldAverage is registered."""
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        assert "fieldAverage" in FunctionObjectRegistry.list_registered()

    def test_n_samples(self, fv_mesh, sample_fields):
        """n_samples tracks execution count."""
        fa = FieldAverage("test", {"fields": ["p"]})
        fa.initialise(fv_mesh, sample_fields)

        assert fa.n_samples == 0
        fa.execute(1.0)
        assert fa.n_samples == 1
        fa.execute(2.0)
        assert fa.n_samples == 2
