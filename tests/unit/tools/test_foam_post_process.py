"""Tests for foam_post_process — post-processing function object runner."""

from __future__ import annotations

import pytest
from pathlib import Path

from pyfoam.tools.foam_post_process import (
    foam_post_process,
    PostProcessResult,
    _discover_times,
)


class TestDiscoverTimes:
    """Test time directory discovery."""

    def test_finds_numeric_directories(self, tmp_path):
        """Numeric directory names are discovered as times."""
        (tmp_path / "0").mkdir()
        (tmp_path / "1").mkdir()
        (tmp_path / "0.5").mkdir()
        (tmp_path / "10").mkdir()
        times = _discover_times(tmp_path)
        assert times == [0.0, 0.5, 1.0, 10.0]

    def test_ignores_non_numeric(self, tmp_path):
        """Non-numeric directory names are ignored."""
        (tmp_path / "0").mkdir()
        (tmp_path / "constant").mkdir()
        (tmp_path / "system").mkdir()
        (tmp_path / "100").mkdir()
        times = _discover_times(tmp_path)
        assert times == [0.0, 100.0]

    def test_empty_case(self, tmp_path):
        """Empty case returns empty list."""
        times = _discover_times(tmp_path)
        assert times == []


class TestFoamPostProcessErrors:
    """Test error handling."""

    def test_missing_case_raises(self, tmp_path):
        """Missing case directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            foam_post_process(tmp_path / "nonexistent", ["forces"])

    def test_no_time_dirs_raises(self, tmp_path):
        """No time directories raises ValueError."""
        with pytest.raises(ValueError, match="No valid time directories"):
            foam_post_process(tmp_path, ["forces"])

    def test_invalid_times_raises(self, tmp_path):
        """Non-existent requested times raise ValueError."""
        (tmp_path / "0").mkdir()
        with pytest.raises(ValueError, match="None of the requested times"):
            foam_post_process(tmp_path, ["forces"], times=[999.0])


class TestFoamPostProcessResult:
    """Test PostProcessResult dataclass."""

    def test_default_result(self):
        """Default result has expected fields."""
        r = PostProcessResult()
        assert r.name == ""
        assert r.type_name == ""
        assert r.data == {}
        assert r.times == []
        assert r.n_fields_read == 0

    def test_custom_result(self):
        """Result can be initialised with data."""
        r = PostProcessResult(
            name="forces1",
            type_name="forces",
            data={"F": [1.0, 2.0]},
            times=[0.0, 1.0],
        )
        assert r.name == "forces1"
        assert r.type_name == "forces"
        assert len(r.data["F"]) == 2


class TestFoamPostProcessGeneric:
    """Test generic (fallback) function object execution."""

    def test_generic_handler(self, tmp_path):
        """Generic handler processes time directories."""
        # Create minimal case with a time dir and a file
        (tmp_path / "0").mkdir()
        (tmp_path / "0" / "p").write_text("placeholder")
        (tmp_path / "0" / "U").write_text("placeholder")

        results = foam_post_process(tmp_path, ["unknownFunction"])
        assert "unknownFunction" in results
        result = results["unknownFunction"]
        assert result.n_fields_read >= 2

    def test_returns_results_dict(self, tmp_path):
        """Function returns dict keyed by function name."""
        (tmp_path / "0").mkdir()
        (tmp_path / "0" / "p").write_text("placeholder")

        results = foam_post_process(tmp_path, ["myFO"])
        assert isinstance(results, dict)
        assert "myFO" in results


class TestFoamPostProcessVolFieldValue:
    """Test volFieldValue function object."""

    def test_vol_average_no_fields(self, tmp_path):
        """volFieldValue with no field data returns empty results."""
        (tmp_path / "0").mkdir()

        results = foam_post_process(tmp_path, ["volFieldValue"])
        assert "volFieldValue" in results

    def test_with_specific_times(self, tmp_path):
        """Processing specific time directories."""
        for t in ["0", "1", "2"]:
            (tmp_path / t).mkdir()
            (tmp_path / t / "p").write_text("placeholder")

        results = foam_post_process(
            tmp_path, ["genericFO"], times=[0.0, 1.0],
        )
        assert "genericFO" in results


class TestFoamPostProcessMultipleFunctions:
    """Test running multiple function objects."""

    def test_multiple_functions(self, tmp_path):
        """Multiple function objects can be run."""
        (tmp_path / "0").mkdir()
        (tmp_path / "0" / "p").write_text("placeholder")

        results = foam_post_process(tmp_path, ["funcA", "funcB"])
        assert "funcA" in results
        assert "funcB" in results
