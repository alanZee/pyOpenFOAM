"""Tests for renumber_mesh_enhanced_3."""
from __future__ import annotations
import pytest
from pyfoam.tools.renumber_mesh_enhanced_3 import RenumberEnhanced3Result, SpaceFillingCurveResult, PartitionAwareResult, renumber_mesh_enhanced_3


class TestRenumberEnhanced3Result:
    def test_returns_result(self):
        r = renumber_mesh_enhanced_3()
        assert isinstance(r, RenumberEnhanced3Result)

    def test_sfc(self):
        r = renumber_mesh_enhanced_3(enable_sfc=True)
        assert isinstance(r.sfc, SpaceFillingCurveResult)
        assert r.sfc.enabled is True

    def test_partition(self):
        r = renumber_mesh_enhanced_3(enable_partition=True)
        assert isinstance(r.partition, PartitionAwareResult)
        assert r.partition.enabled is True
