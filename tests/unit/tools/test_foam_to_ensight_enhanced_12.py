"""Tests for foam_to_ensight_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_to_ensight_enhanced_12 import EnSightV12Result, CloudNativeExportResult, ProvenanceTrackingResult, RealTimeStreamingResult, foam_to_ensight_enhanced_12


class TestEnSightV12Result:
    def test_returns_result(self):
        r = foam_to_ensight_enhanced_12()
        assert isinstance(r, EnSightV12Result)

    def test_cloud(self):
        r = foam_to_ensight_enhanced_12(enable_cloud=True)
        assert isinstance(r.cloud, CloudNativeExportResult)
        assert r.cloud.enabled is True

    def test_provenance(self):
        r = foam_to_ensight_enhanced_12(enable_provenance=True)
        assert isinstance(r.provenance, ProvenanceTrackingResult)
        assert r.provenance.enabled is True

    def test_streaming(self):
        r = foam_to_ensight_enhanced_12(enable_streaming=True)
        assert isinstance(r.streaming, RealTimeStreamingResult)
        assert r.streaming.enabled is True
