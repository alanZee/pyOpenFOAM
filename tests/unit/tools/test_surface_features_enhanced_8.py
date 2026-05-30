"""Tests for surface_features_enhanced_8 — enhanced feature extraction v8."""
from __future__ import annotations
import pytest
import numpy as np
from pyfoam.tools.surface_features_enhanced_8 import (
    SurfaceFeaturesEnhanced8Result, FeatureClassification, TopologyAnalysis,
    MeshConstraint, surface_features_enhanced_8,
)


class TestSurfaceFeatures8:
    def test_returns_result_type_skip(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_features_enhanced_8(vertices=verts, faces=faces)
        assert isinstance(r, SurfaceFeaturesEnhanced8Result)

    def test_classification(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_features_enhanced_8(
            vertices=verts, faces=faces, classify_features=True,
        )
        assert isinstance(r.classifications, list)

    def test_topology(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_features_enhanced_8(
            vertices=verts, faces=faces, analyse_topology=True,
        )
        assert isinstance(r.topology, TopologyAnalysis)

    def test_constraints(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_features_enhanced_8(
            vertices=verts, faces=faces, generate_constraints=True,
        )
        assert isinstance(r.constraints, list)
