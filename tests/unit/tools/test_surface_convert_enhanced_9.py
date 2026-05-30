"""Tests for surface_convert_enhanced_9 — enhanced surface format conversion v9."""
from __future__ import annotations
import pytest
import numpy as np
from pyfoam.tools.surface_convert_enhanced_9 import (
    ConvertEnhanced9Result, MigrationPlan, ConversionValidation,
    IncrementalState, surface_convert_enhanced_9,
)


class TestSurfaceConvertEnhanced9:
    def test_returns_result_type(self):
        r = ConvertEnhanced9Result()
        assert isinstance(r, ConvertEnhanced9Result)

    def test_migration_plan_type(self):
        mp = MigrationPlan(source_format="stl", target_format="ply", n_steps=1)
        assert mp.source_format == "stl"
        assert mp.direct_supported is True

    def test_conversion_validation_type(self):
        cv = ConversionValidation(is_valid=True, n_vertices_match=True)
        assert cv.is_valid is True
        assert cv.max_geometric_error == 0.0

    def test_incremental_type(self):
        inc = IncrementalState(is_incremental=True, n_faces_changed=10)
        assert inc.is_incremental is True
        assert inc.change_ratio == 0.0  # default

    def test_default_values(self):
        r = ConvertEnhanced9Result()
        assert r.n_parallel_inputs == 0
        assert r.incremental.is_incremental is False
