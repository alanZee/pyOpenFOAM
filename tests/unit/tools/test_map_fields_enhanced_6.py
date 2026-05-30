"""Tests for map_fields_enhanced_6."""
from __future__ import annotations
import pytest
from pyfoam.tools.map_fields_enhanced_6 import MapFieldsEnhanced6Result, TetInterpolationResult, MeshToMeshResult, map_fields_enhanced_6


class TestMapFieldsEnhanced6Result:
    def test_returns_result(self):
        r = map_fields_enhanced_6()
        assert isinstance(r, MapFieldsEnhanced6Result)

    def test_tet_interp(self):
        r = map_fields_enhanced_6(enable_tet_interp=True)
        assert isinstance(r.tet_interp, TetInterpolationResult)
        assert r.tet_interp.enabled is True

    def test_mesh_to_mesh(self):
        r = map_fields_enhanced_6(enable_mesh_to_mesh=True)
        assert isinstance(r.mesh_to_mesh, MeshToMeshResult)
        assert r.mesh_to_mesh.enabled is True
