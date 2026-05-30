"""Tests for check_mesh_enhanced_9."""
from __future__ import annotations
import pytest
from pyfoam.tools.check_mesh_enhanced_9 import CheckMeshEnhanced9Result, DistributedCheckResult, AdaptiveThresholdResult, check_mesh_enhanced_9


class TestCheckMeshEnhanced9Result:
    def test_returns_result(self):
        r = check_mesh_enhanced_9()
        assert isinstance(r, CheckMeshEnhanced9Result)

    def test_distributed(self):
        r = check_mesh_enhanced_9(enable_distributed=True)
        assert isinstance(r.distributed, DistributedCheckResult)
        assert r.distributed.enabled is True

    def test_adaptive_threshold(self):
        r = check_mesh_enhanced_9(enable_adaptive_threshold=True)
        assert isinstance(r.adaptive_threshold, AdaptiveThresholdResult)
        assert r.adaptive_threshold.enabled is True
