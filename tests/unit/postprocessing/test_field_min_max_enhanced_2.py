"""Tests for FieldMinMaxEnhanced2.

Tests cover:
- Per-region statistics
- Time derivative tracking
- Convergence monitoring
- Properties and write
"""

import pytest
import torch

from pyfoam.postprocessing.field_min_max_enhanced import FieldMinMaxEnhanced
from pyfoam.postprocessing.field_min_max_enhanced_2 import (
    FieldMinMaxEnhanced2,
    RegionMinMaxResult,
    ConvergenceInfo,
)


class TestFieldMinMaxEnhanced2:
    """Tests for FieldMinMaxEnhanced2."""

    def test_inherits_from_enhanced(self):
        fo = FieldMinMaxEnhanced2("test", {"field": "p"})
        assert isinstance(fo, FieldMinMaxEnhanced)

    def test_default_params(self):
        fo = FieldMinMaxEnhanced2("test", {"field": "p"})
        assert fo._track_convergence is True
        assert fo._convergence_tol == pytest.approx(1e-4)
        assert fo._track_time_derivative is True

    def test_custom_params(self):
        fo = FieldMinMaxEnhanced2("test", {
            "field": "p",
            "regions": ["region0"],
            "convergenceTol": 1e-6,
        })
        assert fo._regions == ["region0"]
        assert fo._convergence_tol == pytest.approx(1e-6)

    def test_execute_with_field(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced2("test", {"field": "p"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(1.0)
        assert len(fo.enhanced_results) == 2

    def test_time_derivative_tracking(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced2("test", {"field": "p", "trackTimeDerivative": True})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(1.0)
        assert len(fo.dmin_dt) == 2
        assert len(fo.dmax_dt) == 2
        # First step has no derivative
        assert fo.dmin_dt[0] == 0.0

    def test_convergence_tracking(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced2("test", {
            "field": "p",
            "trackConvergence": True,
            "convergenceTol": 1.0,  # Large tolerance
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(1.0)
        conv = fo.convergence
        assert isinstance(conv, ConvergenceInfo)
        assert conv.n_steps >= 1

    def test_per_region_analysis(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced2("test", {
            "field": "p",
            "regions": ["bottom"],
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        # Region results should be populated if region matches a patch
        assert isinstance(fo.region_results, dict)

    def test_execute_skips_disabled(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced2("test", {"field": "p", "enabled": False})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.enhanced_results) == 0

    def test_execute_skips_missing_field(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced2("test", {"field": "nonexistent"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.enhanced_results) == 0


# Need conftest fixtures
from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
