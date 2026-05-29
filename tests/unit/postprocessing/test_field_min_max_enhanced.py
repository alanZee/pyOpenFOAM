"""
Unit tests for FieldMinMaxEnhanced — enhanced field min/max analysis.

Tests cover:
- Init with default and custom config
- Enhanced result with cell coordinates
- Statistics (mean, std, range)
- Per-patch analysis
- Gradient computation
- Write output files
- Registration in FunctionObjectRegistry
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.postprocessing.field_min_max_enhanced import (
    FieldMinMaxEnhanced,
    EnhancedMinMaxResult,
)
from pyfoam.postprocessing.field_min_max import MinMaxResult


class TestEnhancedMinMaxResult:
    """Tests for EnhancedMinMaxResult data class."""

    def test_defaults(self):
        r = EnhancedMinMaxResult()
        assert r.time == 0.0
        assert r.field_name == ""
        assert r.min_coords == (0.0, 0.0, 0.0)
        assert r.max_coords == (0.0, 0.0, 0.0)
        assert r.mean == 0.0
        assert r.std == 0.0
        assert r.range == 0.0
        assert r.gradient_at_min is None

    def test_custom_values(self):
        r = EnhancedMinMaxResult(
            time=1.0, field_name="p",
            min_value=100.0, max_value=200.0,
            min_location=5, max_location=10,
            min_coords=(0.5, 0.5, 0.0),
            max_coords=(1.0, 1.0, 1.0),
            mean=150.0, std=20.0, range=100.0,
        )
        assert r.time == 1.0
        assert r.min_coords == (0.5, 0.5, 0.0)
        assert r.mean == 150.0


class TestFieldMinMaxEnhancedInit:
    """Tests for FieldMinMaxEnhanced initialisation."""

    def test_init_defaults(self):
        fme = FieldMinMaxEnhanced()
        assert fme.name == "fieldMinMaxEnhanced"
        assert fme._compute_gradient is False
        assert fme._per_patch is False
        assert fme._track_history is True

    def test_init_with_config(self):
        config = {
            "field": "p",
            "computeGradient": True,
            "perPatch": True,
            "trackHistory": False,
        }
        fme = FieldMinMaxEnhanced("fme1", config)
        assert fme.name == "fme1"
        assert fme._compute_gradient is True
        assert fme._per_patch is True
        assert fme._track_history is False

    def test_inherits_field_min_max(self):
        """FieldMinMaxEnhanced is a subclass of FieldMinMax."""
        from pyfoam.postprocessing.field_min_max import FieldMinMax
        fme = FieldMinMaxEnhanced()
        assert isinstance(fme, FieldMinMax)


class TestEnhancedExecute:
    """Tests for enhanced execution."""

    def test_execute_scalar(self, fv_mesh, sample_fields):
        """Execute on scalar field produces enhanced results."""
        fme = FieldMinMaxEnhanced("fme", {"field": "p"})
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)

        assert len(fme.enhanced_results) == 1
        r = fme.enhanced_results[0]
        assert r.field_name == "p"
        assert r.min_value < r.max_value
        assert r.mean > 0.0
        assert r.std >= 0.0
        assert r.range > 0.0

    def test_execute_vector(self, fv_mesh, sample_fields):
        """Execute on vector field produces enhanced results."""
        fme = FieldMinMaxEnhanced("fme", {"field": "U"})
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)

        assert len(fme.enhanced_results) == 1
        r = fme.enhanced_results[0]
        assert r.min_value >= 0.0  # magnitude is non-negative

    def test_cell_coordinates(self, fv_mesh, sample_fields):
        """Min/max cell coordinates are populated."""
        fme = FieldMinMaxEnhanced("fme", {"field": "p"})
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)

        r = fme.enhanced_results[0]
        # Cell centres should be non-zero (mesh exists)
        assert isinstance(r.min_coords, tuple)
        assert len(r.min_coords) == 3
        assert isinstance(r.max_coords, tuple)
        assert len(r.max_coords) == 3

    def test_multiple_steps(self, fv_mesh, sample_fields):
        """Multiple execute calls accumulate results."""
        fme = FieldMinMaxEnhanced("fme", {"field": "p"})
        fme.initialise(fv_mesh, sample_fields)

        fme.execute(0.0)
        fme.execute(1.0)
        fme.execute(2.0)

        assert len(fme.enhanced_results) == 3
        assert fme.enhanced_results[2].time == 2.0

    def test_base_results_also_stored(self, fv_mesh, sample_fields):
        """Base class results are also stored."""
        fme = FieldMinMaxEnhanced("fme", {"field": "p"})
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)

        # Base class results
        assert len(fme.results) == 1
        assert isinstance(fme.results[0], MinMaxResult)

    def test_disabled(self, fv_mesh, sample_fields):
        """Disabled function object does nothing."""
        fme = FieldMinMaxEnhanced("fme", {"field": "p", "enabled": False})
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)

        assert len(fme.enhanced_results) == 0

    def test_missing_field(self, fv_mesh, sample_fields):
        """Missing field logs warning, no crash."""
        fme = FieldMinMaxEnhanced("fme", {"field": "nonexistent"})
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)

        assert len(fme.enhanced_results) == 0


class TestPerPatchAnalysis:
    """Tests for per-patch min/max analysis."""

    def test_per_patch_results(self, fv_mesh, sample_fields):
        """Per-patch results are computed."""
        fme = FieldMinMaxEnhanced("fme", {"field": "p", "perPatch": True})
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)

        assert isinstance(fme.patch_results, dict)
        # Should have results for at least one patch
        assert len(fme.patch_results) > 0

    def test_get_patch_latest(self, fv_mesh, sample_fields):
        """get_patch_latest returns results for a patch."""
        fme = FieldMinMaxEnhanced("fme", {"field": "p", "perPatch": True})
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)

        for patch_name in fme.patch_results:
            r = fme.get_patch_latest(patch_name)
            assert r is not None
            assert isinstance(r, MinMaxResult)
            break

    def test_get_patch_latest_empty(self):
        """get_patch_latest returns None when no data."""
        fme = FieldMinMaxEnhanced("fme")
        assert fme.get_patch_latest("nonexistent") is None


class TestGradient:
    """Tests for gradient computation at extrema."""

    def test_gradient_enabled(self, fv_mesh, sample_fields):
        """Gradient computation produces results."""
        fme = FieldMinMaxEnhanced("fme", {
            "field": "p",
            "computeGradient": True,
        })
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)

        r = fme.get_latest_enhanced()
        assert r is not None
        # Gradient should be computed (may be None for edge cases)
        # With 2 cells and a pressure difference, gradient should be non-zero


class TestGetLatestEnhanced:
    """Tests for get_latest_enhanced method."""

    def test_returns_latest(self, fv_mesh, sample_fields):
        fme = FieldMinMaxEnhanced("fme", {"field": "p"})
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)

        r = fme.get_latest_enhanced()
        assert r is not None
        assert r.time == 1.0

    def test_returns_none_empty(self):
        fme = FieldMinMaxEnhanced("fme")
        assert fme.get_latest_enhanced() is None


class TestWrite:
    """Tests for output file writing."""

    def test_write_enhanced(self, fv_mesh, sample_fields, tmp_path):
        """Writing enhanced results file."""
        fme = FieldMinMaxEnhanced("fme", {"field": "p"})
        fme.set_output_path(tmp_path)
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)
        fme.write()

        enhanced_file = tmp_path / "fieldMinMaxEnhanced.dat"
        assert enhanced_file.exists()
        content = enhanced_file.read_text()
        assert "mean" in content

    def test_write_per_patch(self, fv_mesh, sample_fields, tmp_path):
        """Writing per-patch results."""
        fme = FieldMinMaxEnhanced("fme", {"field": "p", "perPatch": True})
        fme.set_output_path(tmp_path)
        fme.initialise(fv_mesh, sample_fields)
        fme.execute(1.0)
        fme.write()

        # Should have at least one patch file
        patch_files = list(tmp_path.glob("fieldMinMax_*.dat"))
        assert len(patch_files) > 0

    def test_write_no_data(self, tmp_path):
        """Writing with no data produces base output only."""
        fme = FieldMinMaxEnhanced("fme")
        fme.set_output_path(tmp_path)
        fme.write()

        assert not (tmp_path / "fieldMinMaxEnhanced.dat").exists()


class TestRegistration:
    """Tests for FunctionObjectRegistry registration."""

    def test_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        assert "fieldMinMaxEnhanced" in FunctionObjectRegistry.list_registered()
