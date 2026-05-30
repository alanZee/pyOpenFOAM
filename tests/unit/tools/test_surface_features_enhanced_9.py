"""Tests for surface_features_enhanced_9 — enhanced feature extraction v9."""
from __future__ import annotations
import pytest
import numpy as np
from pyfoam.tools.surface_features_enhanced_9 import (
    SurfaceFeaturesEnhanced9Result, PersistenceRecord, CrossSurfaceCorrelation,
    FeatureExport, surface_features_enhanced_9,
)


class TestSurfaceFeaturesEnhanced9:
    def test_returns_result_type(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_features_enhanced_9(vertices=verts, faces=faces)
        assert isinstance(r, SurfaceFeaturesEnhanced9Result)

    def test_persistence_tracking(self):
        r = surface_features_enhanced_9(
            vertices=np.zeros((10, 3)),
            faces=np.array([[0, 1, 2]]),
            track_persistence=True,
            previous_feature_ids=[0, 1],
            n_iterations=5,
        )
        assert isinstance(r.persistence, list)

    def test_cross_surface_correlation(self):
        other = np.zeros((5, 3))
        r = surface_features_enhanced_9(
            vertices=np.zeros((10, 3)),
            faces=np.array([[0, 1, 2]]),
            correlate_surfaces=[other],
        )
        assert isinstance(r.correlations, list)

    def test_export(self):
        r = surface_features_enhanced_9(
            vertices=np.zeros((10, 3)),
            faces=np.array([[0, 1, 2]]),
            export_format="emesh",
            export_path="test.eMesh",
        )
        assert isinstance(r.export, FeatureExport)
        assert r.export.format == "emesh"

    def test_default_values(self):
        r = SurfaceFeaturesEnhanced9Result()
        assert r.n_features == 0
        assert len(r.persistence) == 0
