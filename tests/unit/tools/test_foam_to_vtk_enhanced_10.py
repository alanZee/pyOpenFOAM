"""Tests for foam_to_vtk_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_to_vtk_enhanced_10 import VtkEnhanced10Result, VtkCloudResult, VtkStreamingResult, VtkMetadataResult, foam_to_vtk_enhanced_10


class TestVtkEnhanced10Result:
    def test_returns_result(self):
        r = foam_to_vtk_enhanced_10()
        assert isinstance(r, VtkEnhanced10Result)

    def test_cloud(self):
        r = foam_to_vtk_enhanced_10(enable_cloud=True)
        assert isinstance(r.cloud, VtkCloudResult)
        assert r.cloud.enabled is True

    def test_streaming(self):
        r = foam_to_vtk_enhanced_10(enable_streaming=True)
        assert isinstance(r.streaming, VtkStreamingResult)
        assert r.streaming.enabled is True

    def test_metadata(self):
        r = foam_to_vtk_enhanced_10(enable_metadata=True)
        assert isinstance(r.metadata, VtkMetadataResult)
        assert r.metadata.enabled is True
