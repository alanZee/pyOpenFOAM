"""Tests for temporal_interpolate_enhanced — enhanced temporal interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from pyfoam.tools.temporal_interpolate_enhanced import (
    temporal_interpolate,
    TemporalInterpolateResult,
)


class TestTemporalInterpolate:
    """Test the temporal_interpolate function."""

    def test_linear_basic(self):
        """Linear interpolation at midpoint should return average."""
        fields = {
            "p": {0.0: np.array([100.0]), 1.0: np.array([200.0])},
        }
        result = temporal_interpolate(fields, [0.5], scheme="linear")
        assert isinstance(result, TemporalInterpolateResult)
        np.testing.assert_allclose(result.fields["p"][0.5], [150.0])

    def test_linear_quarter_point(self):
        """Linear interpolation at t=0.25 should return weighted value."""
        fields = {
            "p": {0.0: np.array([0.0]), 1.0: np.array([100.0])},
        }
        result = temporal_interpolate(fields, [0.25], scheme="linear")
        np.testing.assert_allclose(result.fields["p"][0.25], [25.0])

    def test_linear_at_source_times(self):
        """Interpolation at source times should return exact values."""
        fields = {
            "p": {0.0: np.array([10.0]), 1.0: np.array([20.0])},
        }
        result = temporal_interpolate(fields, [0.0, 1.0], scheme="linear")
        np.testing.assert_allclose(result.fields["p"][0.0], [10.0])
        np.testing.assert_allclose(result.fields["p"][1.0], [20.0])

    def test_cubic_basic(self):
        """Cubic interpolation should work with at least 4 source times."""
        fields = {
            "p": {
                0.0: np.array([0.0]),
                1.0: np.array([1.0]),
                2.0: np.array([4.0]),
                3.0: np.array([9.0]),
            },
        }
        result = temporal_interpolate(fields, [1.5], scheme="cubic")
        val = result.fields["p"][1.5]
        assert np.isfinite(val).all()
        # Should be between 1.0 and 4.0
        assert 0.5 < val[0] < 5.0

    def test_lagrange_basic(self):
        """Lagrange interpolation should work with at least 3 source times."""
        fields = {
            "p": {
                0.0: np.array([0.0]),
                1.0: np.array([1.0]),
                2.0: np.array([4.0]),
            },
        }
        result = temporal_interpolate(fields, [1.5], scheme="lagrange")
        val = result.fields["p"][1.5]
        assert np.isfinite(val).all()

    def test_multiple_fields(self):
        """Multiple fields should all be interpolated."""
        fields = {
            "p": {0.0: np.array([100.0]), 1.0: np.array([200.0])},
            "U": {0.0: np.array([1.0, 0.0, 0.0]), 1.0: np.array([2.0, 0.0, 0.0])},
        }
        result = temporal_interpolate(fields, [0.5], scheme="linear")
        assert result.n_fields == 2
        assert "p" in result.fields
        assert "U" in result.fields
        np.testing.assert_allclose(result.fields["U"][0.5], [1.5, 0.0, 0.0])

    def test_multiple_target_times(self):
        """Multiple target times should all be computed."""
        fields = {
            "p": {0.0: np.array([0.0]), 1.0: np.array([100.0])},
        }
        result = temporal_interpolate(fields, [0.25, 0.5, 0.75], scheme="linear")
        assert len(result.fields["p"]) == 3
        np.testing.assert_allclose(result.fields["p"][0.25], [25.0])
        np.testing.assert_allclose(result.fields["p"][0.75], [75.0])

    def test_derivative_linear(self):
        """Derivative of linear function should be constant."""
        fields = {
            "p": {0.0: np.array([0.0]), 1.0: np.array([100.0])},
        }
        result = temporal_interpolate(fields, [0.5], scheme="linear")
        deriv = result.derivatives["p"][0.5]
        np.testing.assert_allclose(deriv, [100.0])

    def test_extrapolation_clamp(self):
        """Clamp extrapolation should use boundary values."""
        fields = {
            "p": {0.0: np.array([10.0]), 1.0: np.array([20.0])},
        }
        result = temporal_interpolate(
            fields, [-0.5, 1.5], scheme="linear", extrapolation="clamp",
        )
        np.testing.assert_allclose(result.fields["p"][-0.5], [10.0])
        np.testing.assert_allclose(result.fields["p"][1.5], [20.0])

    def test_extrapolation_error(self):
        """Error extrapolation should raise for out-of-range times."""
        fields = {
            "p": {0.0: np.array([10.0]), 1.0: np.array([20.0])},
        }
        with pytest.raises(ValueError, match="outside source range"):
            temporal_interpolate(fields, [-0.5], extrapolation="error")

    def test_extrapolation_linear(self):
        """Linear extrapolation should extend beyond source range."""
        fields = {
            "p": {0.0: np.array([0.0]), 1.0: np.array([100.0])},
        }
        result = temporal_interpolate(
            fields, [2.0], scheme="linear", extrapolation="linear",
        )
        np.testing.assert_allclose(result.fields["p"][2.0], [200.0])

    def test_string_time_keys(self):
        """String time keys should be parsed correctly."""
        fields = {
            "p": {"0": np.array([10.0]), "1": np.array([20.0])},
        }
        result = temporal_interpolate(fields, [0.5], scheme="linear")
        np.testing.assert_allclose(result.fields["p"][0.5], [15.0])

    def test_invalid_scheme_raises(self):
        """Unknown scheme should raise ValueError."""
        fields = {"p": {0.0: np.array([1.0]), 1.0: np.array([2.0])}}
        with pytest.raises(ValueError, match="Unknown scheme"):
            temporal_interpolate(fields, [0.5], scheme="quintic")

    def test_invalid_extrapolation_raises(self):
        """Unknown extrapolation should raise ValueError."""
        fields = {"p": {0.0: np.array([1.0]), 1.0: np.array([2.0])}}
        with pytest.raises(ValueError, match="Unknown extrapolation"):
            temporal_interpolate(fields, [0.5], extrapolation="bounce")

    def test_many_source_times(self):
        """Should handle many source times correctly."""
        n = 20
        times = np.linspace(0, 1, n)
        values = np.sin(2 * np.pi * times)
        fields = {
            "p": {float(t): np.array([v]) for t, v in zip(times, values)},
        }
        result = temporal_interpolate(fields, [0.5], scheme="linear")
        val = result.fields["p"][0.5][0]
        expected = np.sin(np.pi)  # sin(2*pi*0.5) = 0
        assert abs(val - expected) < 0.05

    def test_vector_field_interpolation(self):
        """Vector field interpolation should interpolate each component."""
        fields = {
            "U": {
                0.0: np.array([1.0, 2.0, 3.0]),
                1.0: np.array([4.0, 5.0, 6.0]),
            },
        }
        result = temporal_interpolate(fields, [0.5], scheme="linear")
        np.testing.assert_allclose(
            result.fields["U"][0.5], [2.5, 3.5, 4.5],
        )

    def test_2d_field_interpolation(self):
        """2-D field (matrix) should be interpolated element-wise."""
        fields = {
            "R": {
                0.0: np.array([[1.0, 2.0], [3.0, 4.0]]),
                1.0: np.array([[5.0, 6.0], [7.0, 8.0]]),
            },
        }
        result = temporal_interpolate(fields, [0.5], scheme="linear")
        np.testing.assert_allclose(
            result.fields["R"][0.5], [[3.0, 4.0], [5.0, 6.0]],
        )

    def test_result_metadata(self):
        """Result metadata should be correct."""
        fields = {
            "p": {0.0: np.array([1.0]), 1.0: np.array([2.0])},
        }
        result = temporal_interpolate(fields, [0.25, 0.75], scheme="linear")
        assert result.scheme == "linear"
        assert result.n_fields == 1
        assert result.target_times == [0.25, 0.75]
