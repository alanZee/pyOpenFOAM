"""Tests for merge_meshes_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.merge_meshes_enhanced_12 import MergeEnhanced12Result, DistributedMergeResult, MergeTopologyResult, merge_meshes_enhanced_12


class TestMergeEnhanced12Result:
    def test_returns_result(self):
        r = merge_meshes_enhanced_12()
        assert isinstance(r, MergeEnhanced12Result)

    def test_distributed(self):
        r = merge_meshes_enhanced_12(enable_distributed=True)
        assert isinstance(r.distributed, DistributedMergeResult)
        assert r.distributed.enabled is True

    def test_topology(self):
        r = merge_meshes_enhanced_12(enable_topology=True)
        assert isinstance(r.topology, MergeTopologyResult)
        assert r.topology.enabled is True
