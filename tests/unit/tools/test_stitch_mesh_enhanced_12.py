"""Tests for stitch_mesh_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.stitch_mesh_enhanced_12 import StitchEnhanced12Result, StitchTopologyResult, AutoStitchResult, stitch_mesh_enhanced_12


class TestStitchEnhanced12Result:
    def test_returns_result(self):
        r = stitch_mesh_enhanced_12()
        assert isinstance(r, StitchEnhanced12Result)

    def test_topology(self):
        r = stitch_mesh_enhanced_12(enable_topology=True)
        assert isinstance(r.topology, StitchTopologyResult)
        assert r.topology.enabled is True

    def test_auto_detect(self):
        r = stitch_mesh_enhanced_12(enable_auto_detect=True)
        assert isinstance(r.auto_detect, AutoStitchResult)
        assert r.auto_detect.enabled is True
