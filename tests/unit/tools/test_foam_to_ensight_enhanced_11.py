"""Tests for foam_to_ensight_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_to_ensight_enhanced_11 import EnSightV11Result, AdaptiveExportSchedulingResult, LossyCompressionResult, ExportQualityMetrics, foam_to_ensight_enhanced_11


class TestEnSightV11Result:
    def test_returns_result(self):
        r = foam_to_ensight_enhanced_11()
        assert isinstance(r, EnSightV11Result)

    def test_scheduling(self):
        r = foam_to_ensight_enhanced_11(enable_scheduling=True)
        assert isinstance(r.scheduling, AdaptiveExportSchedulingResult)
        assert r.scheduling.enabled is True

    def test_lossy_compression(self):
        r = foam_to_ensight_enhanced_11(enable_lossy_compression=True)
        assert isinstance(r.lossy_compression, LossyCompressionResult)
        assert r.lossy_compression.enabled is True

    def test_quality_metrics(self):
        r = foam_to_ensight_enhanced_11(enable_quality_metrics=True)
        assert isinstance(r.quality_metrics, ExportQualityMetrics)
        assert r.quality_metrics.enabled is True
