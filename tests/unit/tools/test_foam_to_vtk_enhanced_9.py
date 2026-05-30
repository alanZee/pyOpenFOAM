"""Tests for foam_to_vtk_enhanced_9."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_to_vtk_enhanced_9 import VtkEnhanced9Result, VtkDistributedResult, VtkQualityMetrics, foam_to_vtk_enhanced_9


class TestVtkEnhanced9Result:
    def test_returns_result(self):
        r = foam_to_vtk_enhanced_9()
        assert isinstance(r, VtkEnhanced9Result)

    def test_distributed(self):
        r = foam_to_vtk_enhanced_9(enable_distributed=True)
        assert isinstance(r.distributed, VtkDistributedResult)
        assert r.distributed.enabled is True

    def test_quality(self):
        r = foam_to_vtk_enhanced_9(enable_quality=True)
        assert isinstance(r.quality, VtkQualityMetrics)
        assert r.quality.enabled is True
