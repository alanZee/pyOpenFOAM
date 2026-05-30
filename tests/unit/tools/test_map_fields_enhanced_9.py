"""Tests for map_fields_enhanced_9."""
from __future__ import annotations
import pytest
from pyfoam.tools.map_fields_enhanced_9 import MapFieldsEnhanced9Result, MultiMeshMappingResult, MappingDiagnosticsResult, map_fields_enhanced_9


class TestMapFieldsEnhanced9Result:
    def test_returns_result(self):
        r = map_fields_enhanced_9()
        assert isinstance(r, MapFieldsEnhanced9Result)

    def test_multi_mesh(self):
        r = map_fields_enhanced_9(enable_multi_mesh=True)
        assert isinstance(r.multi_mesh, MultiMeshMappingResult)
        assert r.multi_mesh.enabled is True

    def test_diagnostics(self):
        r = map_fields_enhanced_9(enable_diagnostics=True)
        assert isinstance(r.diagnostics, MappingDiagnosticsResult)
        assert r.diagnostics.enabled is True
