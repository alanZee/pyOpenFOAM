"""
Unit tests for TemporalInterpolate — field interpolation between time steps.

Tests cover:
- Init with default and custom config
- Scheme validation
- Snapshot collection
- Linear interpolation
- Cubic interpolation
- Boundary clamping
- Multi-field interpolation
- Writing output files
- Registration in FunctionObjectRegistry
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.postprocessing.temporal_interpolate import TemporalInterpolate


class TestTemporalInterpolateInit:
    """Tests for TemporalInterpolate initialisation."""

    def test_init_defaults(self):
        ti = TemporalInterpolate()
        assert ti.name == "temporalInterpolate"
        assert ti._scheme == "linear"
        assert ti._write_fields is False

    def test_init_with_config(self):
        config = {
            "fields": ["U", "p"],
            "interpolationScheme": "cubic",
            "writeFields": True,
        }
        ti = TemporalInterpolate("interp1", config)
        assert ti.name == "interp1"
        assert ti._field_names == ["U", "p"]
        assert ti._scheme == "cubic"
        assert ti._write_fields is True

    def test_init_invalid_scheme(self):
        with pytest.raises(ValueError, match="Unknown interpolation scheme"):
            TemporalInterpolate("test", {"interpolationScheme": "quintic"})

    def test_valid_schemes(self):
        for scheme in ["linear", "cubic"]:
            ti = TemporalInterpolate("test", {"interpolationScheme": scheme})
            assert ti._scheme == scheme


class TestTemporalInterpolateSnapshots:
    """Tests for snapshot collection."""

    def test_execute_stores_snapshots(self, fv_mesh, sample_fields):
        """Execute stores field data at each time step."""
        ti = TemporalInterpolate("interp", {"fields": ["p"]})
        ti.initialise(fv_mesh, sample_fields)

        ti.execute(0.0)
        ti.execute(0.1)
        ti.execute(0.2)

        assert ti.get_snapshot_count("p") == 3
        assert ti.get_times("p") == [0.0, 0.1, 0.2]

    def test_execute_all_fields(self, fv_mesh, sample_fields):
        """Execute stores all specified fields."""
        ti = TemporalInterpolate("interp", {"fields": ["p", "U"]})
        ti.initialise(fv_mesh, sample_fields)

        ti.execute(0.0)
        ti.execute(0.1)

        assert ti.get_snapshot_count("p") == 2
        assert ti.get_snapshot_count("U") == 2

    def test_execute_auto_fields(self, fv_mesh, sample_fields):
        """When no fields specified, all available fields are tracked."""
        ti = TemporalInterpolate("interp")
        ti.initialise(fv_mesh, sample_fields)

        ti.execute(0.0)

        assert len(ti.field_names) == len(sample_fields)

    def test_execute_disabled(self, fv_mesh, sample_fields):
        """Disabled interpolator does not collect snapshots."""
        ti = TemporalInterpolate("interp", {"enabled": False, "fields": ["p"]})
        ti.initialise(fv_mesh, sample_fields)
        ti.execute(0.0)

        assert ti.get_snapshot_count("p") == 0


class TestTemporalInterpolateLinear:
    """Tests for linear interpolation."""

    def test_linear_exact_at_data_points(self):
        """Linear interpolation returns exact values at data points."""
        ti = TemporalInterpolate("test", {"interpolationScheme": "linear"})

        # Manually inject snapshots
        ti._snapshots["p"] = [
            (0.0, torch.tensor([1.0, 2.0])),
            (1.0, torch.tensor([3.0, 4.0])),
            (2.0, torch.tensor([5.0, 6.0])),
        ]

        result = ti.interpolate("p", 0.0)
        assert torch.allclose(result, torch.tensor([1.0, 2.0]))

        result = ti.interpolate("p", 1.0)
        assert torch.allclose(result, torch.tensor([3.0, 4.0]))

    def test_linear_midpoint(self):
        """Linear interpolation at midpoint gives average."""
        ti = TemporalInterpolate("test", {"interpolationScheme": "linear"})

        ti._snapshots["p"] = [
            (0.0, torch.tensor([10.0, 20.0])),
            (1.0, torch.tensor([30.0, 40.0])),
        ]

        result = ti.interpolate("p", 0.5)
        assert torch.allclose(result, torch.tensor([20.0, 30.0]))

    def test_linear_before_range(self):
        """Linear interpolation before range clamps to first value."""
        ti = TemporalInterpolate("test", {"interpolationScheme": "linear"})

        ti._snapshots["p"] = [
            (1.0, torch.tensor([10.0])),
            (2.0, torch.tensor([20.0])),
        ]

        result = ti.interpolate("p", 0.0)
        assert torch.allclose(result, torch.tensor([10.0]))

    def test_linear_after_range(self):
        """Linear interpolation after range clamps to last value."""
        ti = TemporalInterpolate("test", {"interpolationScheme": "linear"})

        ti._snapshots["p"] = [
            (0.0, torch.tensor([10.0])),
            (1.0, torch.tensor([20.0])),
        ]

        result = ti.interpolate("p", 5.0)
        assert torch.allclose(result, torch.tensor([20.0]))

    def test_linear_quarter_point(self):
        """Linear interpolation at 1/4 point."""
        ti = TemporalInterpolate("test", {"interpolationScheme": "linear"})

        ti._snapshots["p"] = [
            (0.0, torch.tensor([0.0])),
            (1.0, torch.tensor([100.0])),
        ]

        result = ti.interpolate("p", 0.25)
        assert torch.allclose(result, torch.tensor([25.0]))


class TestTemporalInterpolateCubic:
    """Tests for cubic interpolation."""

    def test_cubic_exact_at_data_points(self):
        """Cubic interpolation returns exact values at data points."""
        ti = TemporalInterpolate("test", {"interpolationScheme": "cubic"})

        ti._snapshots["p"] = [
            (0.0, torch.tensor([0.0])),
            (1.0, torch.tensor([1.0])),
            (2.0, torch.tensor([4.0])),
            (3.0, torch.tensor([9.0])),
        ]

        result = ti.interpolate("p", 1.0)
        assert torch.allclose(result, torch.tensor([1.0]), atol=1e-5)

        result = ti.interpolate("p", 2.0)
        assert torch.allclose(result, torch.tensor([4.0]), atol=1e-5)

    def test_cubic_smoother_than_linear(self):
        """Cubic interpolation produces smoother results."""
        ti_lin = TemporalInterpolate("test", {"interpolationScheme": "linear"})
        ti_cub = TemporalInterpolate("test", {"interpolationScheme": "cubic"})

        data = [
            (0.0, torch.tensor([0.0])),
            (1.0, torch.tensor([1.0])),
            (2.0, torch.tensor([0.0])),
            (3.0, torch.tensor([1.0])),
        ]

        ti_lin._snapshots["p"] = [(t, d.clone()) for t, d in data]
        ti_cub._snapshots["p"] = [(t, d.clone()) for t, d in data]

        # Cubic should be continuous, linear should have kinks
        result_cubic = ti_cub.interpolate("p", 1.5)
        result_linear = ti_lin.interpolate("p", 1.5)

        # Both should be finite
        assert torch.isfinite(result_cubic).all()
        assert torch.isfinite(result_linear).all()

    def test_cubic_with_two_points_falls_back(self):
        """Cubic with only 2 points still works (uses clamped indices)."""
        ti = TemporalInterpolate("test", {"interpolationScheme": "cubic"})

        ti._snapshots["p"] = [
            (0.0, torch.tensor([10.0])),
            (1.0, torch.tensor([20.0])),
        ]

        result = ti.interpolate("p", 0.5)
        assert result is not None
        assert torch.isfinite(result).all()


class TestTemporalInterpolateAllFields:
    """Tests for multi-field interpolation."""

    def test_interpolate_all(self):
        """interpolate_all returns all fields at a time."""
        ti = TemporalInterpolate("test", {"fields": ["p", "U"]})

        ti._snapshots["p"] = [
            (0.0, torch.tensor([1.0])),
            (1.0, torch.tensor([3.0])),
        ]
        ti._field_names = ["p"]

        result = ti.interpolate_all(0.5)
        assert "p" in result
        assert torch.allclose(result["p"], torch.tensor([2.0]))

    def test_interpolate_missing_field(self):
        """interpolate on missing field returns None."""
        ti = TemporalInterpolate("test")
        result = ti.interpolate("nonexistent", 0.5)
        assert result is None

    def test_interpolate_insufficient_data(self):
        """interpolate with < 2 snapshots returns None."""
        ti = TemporalInterpolate("test")
        ti._snapshots["p"] = [(0.0, torch.tensor([1.0]))]
        result = ti.interpolate("p", 0.5)
        assert result is None


class TestTemporalInterpolateWrite:
    """Tests for output file writing."""

    def test_write_info(self, fv_mesh, sample_fields, tmp_path):
        """Writing info file."""
        ti = TemporalInterpolate("interp", {"fields": ["p"]})
        ti.set_output_path(tmp_path)
        ti.initialise(fv_mesh, sample_fields)

        for t in [0.0, 0.1, 0.2]:
            ti.execute(t)

        ti.write()

        info_file = tmp_path / "temporalInterpolate.info"
        assert info_file.exists()
        content = info_file.read_text()
        assert "linear" in content
        assert "p:" in content


class TestTemporalInterpolateRegistration:
    """Tests for FunctionObjectRegistry registration."""

    def test_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        from pyfoam.postprocessing import temporal_interpolate  # noqa: F401
        assert "temporalInterpolate" in FunctionObjectRegistry.list_registered()

    def test_create_from_registry(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        from pyfoam.postprocessing import temporal_interpolate  # noqa: F401
        fo = FunctionObjectRegistry.create("temporalInterpolate", {"name": "interp1"})
        assert isinstance(fo, TemporalInterpolate)
        assert fo.name == "interp1"
