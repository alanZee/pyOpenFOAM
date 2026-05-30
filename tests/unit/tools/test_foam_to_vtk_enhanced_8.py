"""Tests for foam_to_vtk_enhanced_8."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_to_vtk_enhanced_8 import VtkEnhanced8Result, VtkPipelineResult, PolyhedraResult, foam_to_vtk_enhanced_8


class TestVtkEnhanced8Result:
    def test_returns_result(self):
        r = foam_to_vtk_enhanced_8()
        assert isinstance(r, VtkEnhanced8Result)

    def test_pipeline(self):
        r = foam_to_vtk_enhanced_8(enable_pipeline=True)
        assert isinstance(r.pipeline, VtkPipelineResult)
        assert r.pipeline.enabled is True

    def test_polyhedra(self):
        r = foam_to_vtk_enhanced_8(enable_polyhedra=True)
        assert isinstance(r.polyhedra, PolyhedraResult)
        assert r.polyhedra.enabled is True
